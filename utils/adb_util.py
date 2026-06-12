"""Autonomous Database management UI utilities."""

import logging
import os
import zipfile
from pathlib import Path

import gradio as gr
import oci
import pandas as pd
from dotenv import find_dotenv

logger = logging.getLogger(__name__)

OCI_CONFIG_PATH = "/root/.oci/config"
DEFAULT_REGIONS = ["ap-osaka-1", "ap-tokyo-1", "us-chicago-1"]


def _oci_config_path() -> str:
    if Path(OCI_CONFIG_PATH).exists():
        return OCI_CONFIG_PATH
    found = find_dotenv(OCI_CONFIG_PATH)
    return found or OCI_CONFIG_PATH


def _default_region() -> str:
    return DEFAULT_REGIONS[0]


def _region_choices() -> list[str]:
    choices = [DEFAULT_REGIONS[0]]
    env_region = os.environ.get("OCI_REGION", "").strip()
    if env_region:
        choices.append(env_region)
    try:
        cfg_path = _oci_config_path()
        if Path(cfg_path).exists():
            cfg = oci.config.from_file(file_location=cfg_path)
            cfg_region = str(cfg.get("region") or "").strip()
            if cfg_region:
                choices.append(cfg_region)
    except Exception as e:
        logger.warning(f"Unable to read OCI region from config: {e}")
    choices.extend(DEFAULT_REGIONS)
    return list(dict.fromkeys([r for r in choices if r]))


def _oci_config_with_region(region: str) -> dict:
    cfg_path = _oci_config_path()
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"OCI config was not found: {cfg_path}")
    cfg = oci.config.from_file(file_location=cfg_path)
    if region:
        cfg["region"] = region
    return cfg


def _get_adb(client: oci.database.DatabaseClient, adb_id: str):
    return client.get_autonomous_database(autonomous_database_id=adb_id).data


def _start_adb(region: str, adb_id: str):
    cfg = _oci_config_with_region(region)
    client = oci.database.DatabaseClient(cfg)
    client.start_autonomous_database(autonomous_database_id=adb_id)
    try:
        return _get_adb(client, adb_id)
    except Exception as e:
        logger.error(f"_start_adb status fetch error: {e}")
        return None


def _stop_adb(region: str, adb_id: str):
    cfg = _oci_config_with_region(region)
    client = oci.database.DatabaseClient(cfg)
    client.stop_autonomous_database(autonomous_database_id=adb_id)
    try:
        return _get_adb(client, adb_id)
    except Exception as e:
        logger.error(f"_stop_adb status fetch error: {e}")
        return None


def _wallet_dir() -> Path:
    lib_dir = os.environ.get("ORACLE_CLIENT_LIB_DIR", "/u01/aipoc/instantclient_23_26")
    return Path(lib_dir) / "network" / "admin"


def _check_wallet_files(wallet_dir: Path) -> bool:
    required_files = ["cwallet.sso", "sqlnet.ora", "tnsnames.ora"]
    if not wallet_dir.exists():
        return False
    return all((wallet_dir / name).exists() for name in required_files)


def _fix_sqlnet(wallet_dir: Path):
    sqlnet_path = wallet_dir / "sqlnet.ora"
    if not sqlnet_path.exists():
        return
    try:
        content = sqlnet_path.read_text(encoding="utf-8", errors="ignore")
        updated = content.replace(
            'DIRECTORY="?/network/admin"', f'DIRECTORY="{wallet_dir}"'
        )
        if updated != content:
            sqlnet_path.write_text(updated, encoding="utf-8")
    except Exception as e:
        logger.error(f"sqlnet.ora update error: {e}")


def _cleanup_wallet(wallet_dir: Path):
    for name in [
        "README",
        "keystore.jks",
        "truststore.jks",
        "ojdbc.properties",
        "ewallet.p12",
    ]:
        try:
            p = wallet_dir / name
            if p.exists():
                p.unlink()
        except Exception as e:
            logger.error(f"wallet cleanup error ({name}): {e}")


def _download_and_extract_wallet(
    region: str, adb_ocid: str, wallet_password: str = "WalletPassword123"
) -> bool:
    try:
        cfg = _oci_config_with_region(region)
        client = oci.database.DatabaseClient(cfg)
        wallet_details = oci.database.models.GenerateAutonomousDatabaseWalletDetails(
            password=wallet_password
        )
        wallet_response = client.generate_autonomous_database_wallet(
            autonomous_database_id=adb_ocid,
            generate_autonomous_database_wallet_details=wallet_details,
        )

        wallet_zip_path = Path("/tmp/no1-rag-wallet.zip")
        with wallet_zip_path.open("wb") as f:
            for chunk in wallet_response.data.raw.stream(
                1024 * 1024, decode_content=False
            ):
                f.write(chunk)

        wallet_dir = _wallet_dir()
        wallet_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(wallet_zip_path, "r") as zip_ref:
            zip_ref.extractall(wallet_dir)

        _fix_sqlnet(wallet_dir)
        _cleanup_wallet(wallet_dir)
        return True
    except Exception as e:
        logger.error(f"Wallet download/extract error: {e}")
        return False


def _state_to_val(value):
    try:
        if isinstance(value, dict):
            return value
        state_value = getattr(value, "value", None)
        if isinstance(state_value, dict):
            return state_value
    except Exception as e:
        logger.error(f"_state_to_val error: {e}")
    return {}


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["表示名", "状態", "OCID"])


def _mp_to_df(mp: dict) -> pd.DataFrame:
    rows = [[v.get("name"), v.get("state"), ocid] for ocid, v in (mp or {}).items()]
    return pd.DataFrame(rows, columns=["表示名", "状態", "OCID"]) if rows else _empty_df()


def _df_from_gradio_value(value) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, dict):
        headers = value.get("headers") or value.get("column_names") or []
        data = value.get("data") or []
        return pd.DataFrame(data, columns=headers if headers else None)
    return pd.DataFrame(value)


def _buttons_for_state(state: str):
    can_start = state in ("STOPPED", "INACTIVE")
    can_stop = state in ("AVAILABLE", "RUNNING", "STARTING")
    return gr.Button(interactive=can_start), gr.Button(interactive=can_stop)


def build_adb_tab(pool=None):
    default_region = _default_region()

    with gr.Accordion(label="", open=True):
        with gr.Column(scale=1):
            region_input = gr.Dropdown(
                choices=_region_choices(),
                value=default_region,
                label="Region*",
                interactive=True,
            )
            with gr.Row():
                fetch_btn = gr.Button(value="ADB情報を取得", variant="primary")

        adb_status_md = gr.Markdown(visible=False)
        adb_list_df = gr.Dataframe(
            label="ADB情報(件数: 0)",
            interactive=False,
            wrap=True,
            visible=False,
            value=_empty_df(),
        )

        with gr.Row():
            start_btn = gr.Button(value="起動", interactive=False, variant="primary")
            stop_btn = gr.Button(value="停止", interactive=False, variant="primary")

        btn_status_md = gr.Markdown(visible=False)
        adb_map_state = gr.State({})
        adb_selected_id = gr.State("")

        def _fetch(region):
            adb_ocid = os.environ.get("ADB_OCID", "").strip()
            if not adb_ocid or adb_ocid == "ocid1.autonomousdatabase.oc1..":
                yield (
                    gr.Markdown(visible=True, value="ADB_OCIDが設定されていません"),
                    gr.Dataframe(visible=False, value=_empty_df(), label="ADB情報(件数: 0)"),
                    {},
                    "",
                    gr.Button(interactive=False),
                    gr.Button(interactive=False),
                )
                return

            yield (
                gr.Markdown(visible=True, value="ADB情報を取得中..."),
                gr.Dataframe(visible=False, value=_empty_df(), label="ADB情報(件数: 0)"),
                {},
                "",
                gr.Button(interactive=False),
                gr.Button(interactive=False),
            )

            wallet_status = ""
            wallet_dir = _wallet_dir()
            if not _check_wallet_files(wallet_dir):
                wallet_status = "\nWalletファイルが見つかりません。ダウンロード中..."
                yield (
                    gr.Markdown(visible=True, value=f"ADB情報を取得中...{wallet_status}"),
                    gr.Dataframe(visible=False, value=_empty_df(), label="ADB情報(件数: 0)"),
                    {},
                    "",
                    gr.Button(interactive=False),
                    gr.Button(interactive=False),
                )
                if _download_and_extract_wallet(region, adb_ocid):
                    wallet_status = "\nWalletファイルをダウンロードしました"
                else:
                    yield (
                        gr.Markdown(visible=True, value="Walletファイルのダウンロードに失敗しました"),
                        gr.Dataframe(visible=False, value=_empty_df(), label="ADB情報(件数: 0)"),
                        {},
                        "",
                        gr.Button(interactive=False),
                        gr.Button(interactive=False),
                    )
                    return
            else:
                wallet_status = "\nWalletファイルは既に存在します"

            try:
                cfg = _oci_config_with_region(region)
                client = oci.database.DatabaseClient(cfg)
                adb = _get_adb(client, adb_ocid)
                name = adb.display_name
                state = adb.lifecycle_state
                oid = adb.id
                mp = {oid: {"name": name, "state": state}}
                df = _mp_to_df(mp)
                start_update, stop_update = _buttons_for_state(state)
                yield (
                    gr.Markdown(visible=True, value=f"取得完了{wallet_status}"),
                    gr.Dataframe(visible=True, value=df, label=f"ADB情報(件数: {len(df)})"),
                    mp,
                    oid,
                    start_update,
                    stop_update,
                )
            except Exception as e:
                logger.error(f"_fetch error: {e}")
                yield (
                    gr.Markdown(visible=True, value=f"エラー: {e}"),
                    gr.Dataframe(visible=False, value=_empty_df(), label="ADB情報(件数: 0)"),
                    {},
                    "",
                    gr.Button(interactive=False),
                    gr.Button(interactive=False),
                )

        def _on_row_select(evt: gr.SelectData, current_df, mp):
            try:
                df = _df_from_gradio_value(current_df)
                row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                if len(df) <= row_index:
                    raise IndexError(row_index)
                ocid = str(df.iloc[row_index]["OCID"])
                name = str(df.iloc[row_index]["表示名"])
                state = str(df.iloc[row_index]["状態"])
                info = _state_to_val(mp).get(ocid) or {}
                state = info.get("state") or state
                start_update, stop_update = _buttons_for_state(state)
                return (
                    gr.Markdown(visible=True, value=f"選択: {name} / 状態: {state}"),
                    start_update,
                    stop_update,
                    ocid,
                )
            except Exception as e:
                logger.error(f"_on_row_select error: {e}")
                return (
                    gr.Markdown(visible=True, value="行をクリックしてADBを選択してください"),
                    gr.Button(interactive=False),
                    gr.Button(interactive=False),
                    "",
                )

        def _start(region, selected_id, mp):
            mpv = _state_to_val(mp)
            if not selected_id:
                yield (
                    gr.Markdown(visible=True, value="ADBが選択されていません"),
                    gr.Button(interactive=False),
                    gr.Button(interactive=False),
                    mpv,
                    gr.Dataframe(visible=True, value=_mp_to_df(mpv), label=f"ADB情報(件数: {len(mpv)})"),
                )
                return

            yield (
                gr.Markdown(visible=True, value="起動をリクエスト中..."),
                gr.Button(interactive=False),
                gr.Button(interactive=False),
                mpv,
                gr.Dataframe(visible=True, value=_mp_to_df(mpv), label=f"ADB情報(件数: {len(mpv)})"),
            )
            try:
                adb = _start_adb(region, selected_id)
                state = getattr(adb, "lifecycle_state", None) or "STARTING"
                name = mpv.get(selected_id, {}).get("name", selected_id)
                mpv[selected_id] = {"name": name, "state": state}
                yield (
                    gr.Markdown(visible=True, value="起動リクエストを送信しました。数分後にADB情報を再取得してください。"),
                    gr.Button(interactive=False),
                    gr.Button(interactive=True),
                    mpv,
                    gr.Dataframe(visible=True, value=_mp_to_df(mpv), label=f"ADB情報(件数: {len(mpv)})"),
                )
            except Exception as e:
                logger.error(f"_start error: {e}")
                yield (
                    gr.Markdown(visible=True, value=f"起動エラー: {e}"),
                    gr.Button(interactive=True),
                    gr.Button(interactive=False),
                    mpv,
                    gr.Dataframe(visible=True, value=_mp_to_df(mpv), label=f"ADB情報(件数: {len(mpv)})"),
                )

        def _stop(region, selected_id, mp):
            mpv = _state_to_val(mp)
            if not selected_id:
                yield (
                    gr.Markdown(visible=True, value="ADBが選択されていません"),
                    gr.Button(interactive=False),
                    gr.Button(interactive=False),
                    mpv,
                    gr.Dataframe(visible=True, value=_mp_to_df(mpv), label=f"ADB情報(件数: {len(mpv)})"),
                )
                return

            yield (
                gr.Markdown(visible=True, value="停止をリクエスト中..."),
                gr.Button(interactive=False),
                gr.Button(interactive=False),
                mpv,
                gr.Dataframe(visible=True, value=_mp_to_df(mpv), label=f"ADB情報(件数: {len(mpv)})"),
            )
            try:
                if pool is not None:
                    try:
                        pool.close()
                    except Exception as e:
                        logger.error(f"pool.close during ADB stop error: {e}")
                adb = _stop_adb(region, selected_id)
                state = getattr(adb, "lifecycle_state", None) or "STOPPING"
                name = mpv.get(selected_id, {}).get("name", selected_id)
                mpv[selected_id] = {"name": name, "state": state}
                yield (
                    gr.Markdown(visible=True, value="停止リクエストを送信しました。数分後にADB情報を再取得してください。"),
                    gr.Button(interactive=False),
                    gr.Button(interactive=False),
                    mpv,
                    gr.Dataframe(visible=True, value=_mp_to_df(mpv), label=f"ADB情報(件数: {len(mpv)})"),
                )
            except Exception as e:
                logger.error(f"_stop error: {e}")
                yield (
                    gr.Markdown(visible=True, value=f"停止エラー: {e}"),
                    gr.Button(interactive=False),
                    gr.Button(interactive=True),
                    mpv,
                    gr.Dataframe(visible=True, value=_mp_to_df(mpv), label=f"ADB情報(件数: {len(mpv)})"),
                )

        fetch_btn.click(
            _fetch,
            inputs=[region_input],
            outputs=[
                adb_status_md,
                adb_list_df,
                adb_map_state,
                adb_selected_id,
                start_btn,
                stop_btn,
            ],
        )
        adb_list_df.select(
            _on_row_select,
            inputs=[adb_list_df, adb_map_state],
            outputs=[adb_status_md, start_btn, stop_btn, adb_selected_id],
        )
        start_btn.click(
            _start,
            inputs=[region_input, adb_selected_id, adb_map_state],
            outputs=[btn_status_md, start_btn, stop_btn, adb_map_state, adb_list_df],
        )
        stop_btn.click(
            _stop,
            inputs=[region_input, adb_selected_id, adb_map_state],
            outputs=[btn_status_md, start_btn, stop_btn, adb_map_state, adb_list_df],
        )
