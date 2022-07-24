import os
from surqct.pipelines import codes

codes_dir = os.path.dirname(codes.__file__)
__all__ = {
    'MCoarseAir': codes_dir + '/MCoarseAir/Exec/',
    'Plato': codes_dir + '/Plato/Exec/'
}
