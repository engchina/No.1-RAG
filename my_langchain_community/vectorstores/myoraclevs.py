from __future__ import annotations

import array
import functools
import hashlib
import json
import logging
import os
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import oracledb
from langchain_core.documents import Document

if TYPE_CHECKING:
    from oracledb import Connection

from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores import OracleVS
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
)

logger = logging.getLogger(__name__)
log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Define a type variable that can be any kind of function
T = TypeVar("T", bound=Callable[..., Any])

_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain_oracle"


def _handle_exceptions(func: T) -> T:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except RuntimeError as db_err:
            # Handle a known type of error (e.g., DB-related) specifically
            logger.exception("DB-related error occurred.")
            raise RuntimeError(
                "Failed due to a DB issue: {}".format(db_err)
            ) from db_err
        except ValueError as val_err:
            # Handle another known type of error specifically
            logger.exception("Validation error.")
            raise ValueError("Validation failed: {}".format(val_err)) from val_err
        except Exception as e:
            # Generic handler for all other exceptions
            logger.exception("An unexpected error occurred: {}".format(e))
            raise RuntimeError("Unexpected error: {}".format(e)) from e

    return cast(T, wrapper)


class MyOracleVS(OracleVS):
    """`OracleVS` vector store.

    To use, you should have both:
    - the ``oracledb`` python package installed
    - a connection string associated with a OracleDBCluster having deployed an
       Search index
    """

    def __init__(
        self,
        client: Connection,
        embedding_function: Union[
            Callable[[str], List[float]],
            Embeddings,
        ],
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[dict] = None,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        query: Optional[str] = "What is a Oracle database",
        params: Optional[Dict[str, Any]] = None,
    ):
        try:
            import oracledb
        except ImportError as e:
            raise ImportError(
                "Unable to import oracledb, please install with "
                "`pip install -U oracledb`."
            ) from e

        try:
            """Initialize with oracledb client."""
            self.client = client
            """Initialize with necessary components."""
            if not isinstance(embedding_function, Embeddings):
                logger.warning(
                    "`embedding_function` is expected to be an Embeddings "
                    "object, support "
                    "for passing in a function will soon be removed."
                )
            self.embedding_function = embedding_function
            self.query = query
            embedding_dim = self.get_embedding_dimension()

            self.collection_name = collection_name
            self.collection_metadata = collection_metadata
            self.distance_strategy = distance_strategy
            self.params = params

            pre_delete_collection = params.get("pre_delete_collection", False)
            if pre_delete_collection:
                drop_table_purge(client, collection_name)

            _create_table(client, collection_name, embedding_dim)
        except oracledb.DatabaseError as db_err:
            logger.exception(f"Database error occurred while create table: {db_err}")
            raise RuntimeError(
                "Failed to create table due to a database error."
            ) from db_err
        except ValueError as val_err:
            logger.exception(f"Validation error: {val_err}")
            raise RuntimeError(
                "Failed to create table due to a validation error."
            ) from val_err
        except Exception as ex:
            logger.exception("An unexpected error occurred while creating the index.")
            raise RuntimeError(
                "Failed to create table due to an unexpected error."
            ) from ex


@_handle_exceptions
def drop_table_purge(client: Connection, collection_name: str) -> None:
    if _table_exists(client, collection_name + "_embedding"):
        cursor = client.cursor()
        with cursor:
            drop_embedding_ddl = f"DROP TABLE {collection_name}_embedding PURGE"
            cursor.execute(drop_embedding_ddl)
        logger.info("Table dropped successfully...")
    else:
        logger.info("Table not found...")
    if _table_exists(client, collection_name + "_collection"):
        cursor = client.cursor()
        with cursor:
            drop_collection_ddl = f"DROP TABLE {collection_name}_collection PURGE"
            cursor.execute(drop_collection_ddl)
        logger.info("Table dropped successfully...")
    else:
        logger.info("Table not found...")
    return


def _table_exists(client: Connection, table_name: str) -> bool:
    try:
        import oracledb
    except ImportError as e:
        raise ImportError(
            "Unable to import oracledb, please install with "
            "`pip install -U oracledb`."
        ) from e

    try:
        with client.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return True
    except oracledb.DatabaseError as ex:
        err_obj = ex.args
        if err_obj[0].code == 942:
            return False
        raise


def _get_distance_function(distance_strategy: DistanceStrategy) -> str:
    # Dictionary to map distance strategies to their corresponding function
    # names
    distance_strategy2function = {
        DistanceStrategy.EUCLIDEAN_DISTANCE: "EUCLIDEAN",
        DistanceStrategy.DOT_PRODUCT: "DOT",
        DistanceStrategy.COSINE: "COSINE",
    }

    # Attempt to return the corresponding distance function
    if distance_strategy in distance_strategy2function:
        return distance_strategy2function[distance_strategy]

    # If it's an unsupported distance strategy, raise an error
    raise ValueError(f"Unsupported distance strategy: {distance_strategy}")


@_handle_exceptions
def _create_table(client: Connection, collection_name: str, embedding_dim: int) -> None:
    collection_cols_dict = {
        "id": "VARCHAR2(200)",
        "data": "BLOB",
        "cmetadata": "CLOB"
    }

    if not _table_exists(client, collection_name + "_collection"):
        with client.cursor() as cursor:
            ddl_body = ", ".join(
                f"{col_name} {col_type}" for col_name, col_type in collection_cols_dict.items()
            )
            ddl = f"CREATE TABLE IF NOT EXISTS {collection_name}_collection ({ddl_body})"
            cursor.execute(ddl)
        logger.info(f"Table {collection_name}_collection created successfully...")
    else:
        logger.info(f"Table {collection_name}_collection already exists...")

    embedding_cols_dict = {
        "doc_id": "VARCHAR2(200)",
        "embed_id": "NUMBER",
        "embed_data": "VARCHAR2(2000)",
        "embed_vector": f"vector({embedding_dim}, FLOAT32)",
        "cmetadata": "CLOB"
    }

    if not _table_exists(client, collection_name + "_embedding"):
        with client.cursor() as cursor:
            ddl_body = ", ".join(
                f"{col_name} {col_type}" for col_name, col_type in embedding_cols_dict.items()
            )
            ddl = f"CREATE TABLE IF NOT EXISTS {collection_name}_embedding ({ddl_body})"
            cursor.execute(ddl)
        logger.info(f"Table {collection_name}_embedding created successfully...")
    else:
        logger.info(f"Table {collection_name}_embedding already exists...")

    @_handle_exceptions
    def add_texts(
        self,
        embed_datas: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add more embed_datas to the vectorstore index.
        Args:
          embed_datas: Iterable of strings to add to the vectorstore.
          metadatas: Optional list of metadatas associated with the embed_datas.
          ids: Optional list of ids for the embed_datas that are being added to
          the vector store.
          kwargs: vectorstore specific parameters
        """

        # doc_id = str(uuid.uuid4())
        print(f"{kwargs=}")
        doc_id = kwargs.get("doc_id", str(uuid.uuid4()))
        embed_datas = list(embed_datas)
        # Generate new ids if none are provided
        processed_ids = [i for i in range(1, len(embed_datas) + 1)]

        embeddings = self._embed_documents(embed_datas)
        if not metadatas:
            metadatas = [{} for _ in embed_datas]
        docs = [
            (doc_id, embed_id, embed_data, json.dumps(metadata), array.array("f", embedding))
            for embed_id, embed_data, metadata, embedding in zip(
                processed_ids, embed_datas, metadatas, embeddings
            )
        ]

        with self.client.cursor() as cursor:
            cursor.executemany(
                f"INSERT INTO {self.collection_name}_embedding (doc_id, embed_id, embed_data, cmetadata, "
                f"embed_vector) VALUES (:1, :2, :3, :4, :5)",
                docs,
            )
            self.client.commit()
        return [str(i) for i in processed_ids]

    @_handle_exceptions
    def _get_clob_value(self, result: Any) -> str:
        try:
            import oracledb
        except ImportError as e:
            raise ImportError(
                "Unable to import oracledb, please install with "
                "`pip install -U oracledb`."
            ) from e

        clob_value = ""
        if result:
            if isinstance(result, oracledb.LOB):
                raw_data = result.read()
                if isinstance(raw_data, bytes):
                    clob_value = raw_data.decode(
                        "utf-8"
                    )  # Specify the correct encoding
                else:
                    clob_value = raw_data
            elif isinstance(result, str):
                clob_value = result
            else:
                raise Exception("Unexpected type:", type(result))
        return clob_value

    @_handle_exceptions
    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        docs_and_scores = []
        # embedding_arr =  array.array("f", embedding)
        inputsizes_parameters = {"query_embedding": oracledb.DB_TYPE_VECTOR}
        keyword_parameters = {"query_embedding": embedding}
        similarity_threshold = kwargs.get("similarity_threshold", 0.95)
        query = f"""(
            SELECT embed_id,
              embed_data,
              cmetadata,
              vector_distance(embed_vector, :query_embedding,
              {_get_distance_function(self.distance_strategy)}) as distance,
              doc_id
            FROM {self.collection_name}_embedding
            WHERE vector_distance(embed_vector, :query_embedding,
              {_get_distance_function(self.distance_strategy)}) <= {similarity_threshold} )
            ORDER BY distance
            FETCH APPROX FIRST {k} ROWS ONLY """

        # Execute the query
        with self.client.cursor() as cursor:
            # print(f"{query=}")
            print(f"{inputsizes_parameters=}")
            print(f"{keyword_parameters=}")
            cursor.setinputsizes(**inputsizes_parameters)
            cursor.execute(query, **keyword_parameters)
            results = cursor.fetchall()

            # Filter results if filter is provided
            for result in results:
                metadata = json.loads(
                    self._get_clob_value(result[2]) if result[2] is not None else "{}"
                )

                # Apply filtering based on the 'filter' dictionary
                if filter:
                    if all(metadata.get(key) in value for key, value in filter.items()):
                        doc = Document(
                            page_content=(
                                self._get_clob_value(result[1])
                                if result[1] is not None
                                else ""
                            ),
                            metadata=metadata,
                        )
                        distance = result[3]
                        docs_and_scores.append((doc, distance))
                else:
                    doc = Document(
                        page_content=(
                            self._get_clob_value(result[1])
                            if result[1] is not None
                            else ""
                        ),
                        metadata=metadata,
                    )
                    distance = result[3]
                    docs_and_scores.append((doc, distance))

        return docs_and_scores

    @_handle_exceptions
    def similarity_search_by_vector_returning_embeddings(
        self,
        embedding: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float, np.ndarray[np.float32, Any]]]:
        documents = []
        embedding_arr = array.array("f", embedding)

        query = f"""
        SELECT id,
          text,
          metadata,
          vector_distance(embedding, :embedding, {_get_distance_function(
            self.distance_strategy)}) as distance,
          embedding
        FROM {self.collection_name}
        ORDER BY distance
        FETCH APPROX FIRST {k} ROWS ONLY
        """

        # Execute the query
        with self.client.cursor() as cursor:
            cursor.execute(query, embedding=embedding_arr)
            results = cursor.fetchall()

            for result in results:
                page_content_str = self._get_clob_value(result[1])
                metadata_str = self._get_clob_value(result[2])
                metadata = json.loads(metadata_str)

                # Apply filter if provided and matches; otherwise, add all
                # documents
                if not filter or all(
                    metadata.get(key) in value for key, value in filter.items()
                ):
                    document = Document(
                        page_content=page_content_str, metadata=metadata
                    )
                    distance = result[3]
                    # Assuming result[4] is already in the correct format;
                    # adjust if necessary
                    current_embedding = (
                        np.array(result[4], dtype=np.float32)
                        if result[4]
                        else np.empty(0, dtype=np.float32)
                    )
                    documents.append((document, distance, current_embedding))
        return documents  # type: ignore

    @_handle_exceptions
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete by vector IDs.
        Args:
          self: An instance of the class
          ids: List of ids to delete.
          **kwargs
        """

        if ids is None:
            raise ValueError("No ids provided to delete.")

        # Compute SHA-256 hashes of the ids and truncate them
        hashed_ids = [
            hashlib.sha256(_id.encode()).hexdigest()[:16].upper() for _id in ids
        ]

        # Constructing the SQL statement with individual placeholders
        placeholders = ", ".join([":id" + str(i + 1) for i in range(len(hashed_ids))])

        ddl = f"DELETE FROM {self.collection_name} WHERE id IN ({placeholders})"

        # Preparing bind variables
        bind_vars = {
            f"id{i}": hashed_id for i, hashed_id in enumerate(hashed_ids, start=1)
        }

        with self.client.cursor() as cursor:
            cursor.execute(ddl, bind_vars)
            self.client.commit()

    @classmethod
    @_handle_exceptions
    def from_texts(
        cls: Type[MyOracleVS],
        embed_datas: Iterable[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> MyOracleVS:
        """Return VectorStore initialized from embed_datas and embeddings."""
        client = kwargs.get("client")
        if client is None:
            raise ValueError("client parameter is required...")
        params = kwargs.get("params", {})

        collection_name = str(kwargs.get("collection_name", _LANGCHAIN_DEFAULT_COLLECTION_NAME))

        distance_strategy = cast(
            DistanceStrategy, kwargs.get("distance_strategy", None)
        )
        if not isinstance(distance_strategy, DistanceStrategy):
            raise TypeError(
                f"Expected DistanceStrategy got " f"{type(distance_strategy).__name__} "
            )

        query = kwargs.get("query", "What is a Oracle database")

        pre_delete_collection = kwargs.get("pre_delete_collection")
        if pre_delete_collection:
            drop_table_purge(client, collection_name)

        vss = cls(
            client=client,
            embedding_function=embedding,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            query=query,
            params=params,
        )
        vss.add_texts(embed_datas=list(embed_datas), metadatas=metadatas, **kwargs)
        return vss
