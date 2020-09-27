from typing import Dict, Any, Optional


def flatten_dictionary(dictionary: Dict[str, Any],
                       separator: str = '.',
                       prefix: Optional[str] = '',
                       ) -> Dict[str, Any]:
    flat_dict = {prefix + separator + k if prefix else k: v
                 for kk, vv in dictionary.items()
                 for k, v in flatten_dictionary(vv, separator, kk).items()
                 } if isinstance(dictionary, dict) else {prefix: dictionary}
    return flat_dict
