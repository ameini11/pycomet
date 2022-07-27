import os
from surqct.pipelines import codes

codes_dir = os.path.dirname(codes.__file__)
__all__ = {
    'MCoarseAIR': codes_dir + '/MCoarseAIR/Exec/',
    'Plato': codes_dir + '/Plato/Exec/'
}
