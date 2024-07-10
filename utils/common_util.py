def get_dict_value(dictionary, key, default_value=None):
    """
    Safely get a value from a dictionary.

    Args:
    dictionary (dict): The dictionary to get the value from.
    key: The key to look up in the dictionary.
    default_value: The value to return if the key is not found. Defaults to None.

    Returns:
    The value associated with the key if it exists, otherwise the default_value.
    """
    try:
        return dictionary[key]
    except KeyError:
        return default_value
