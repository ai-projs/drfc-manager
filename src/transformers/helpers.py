from typing import TypeVar, Any

from gloe import transformer, partial_transformer, Transformer
from gloe.utils import forward, debug, forward_incoming

_In = TypeVar('_In')

# side_effect


def side_effect(_transformer: Transformer[_In, Any]) -> Transformer[_In, _In]:
    """
    This transformer just executes another one but does not pass
     anything to the next transformer on pipeline execution
    """
    
    @transformer
    def _pick_income(income: tuple[Any, _In]) -> _In:
        return income[1]
        
    return forward_incoming(_transformer) >> _pick_income
    
