from dmv.models.multi.dlf import Mean as FusionMean, MeanPost as FusionMeanPost
from dmv.models.multi.dynamic_dense import Mean as DDMean, MeanTanh as DDMeanTanh, MeanStd as DDMeanStd, MeanStdTanh as DDMeanStdTanh
from dmv.models.single import Mura

SINGLE_MODELS = [Mura]
MULTI_MODELS = [FusionMean, FusionMeanPost, DDMean, DDMeanTanh, DDMeanStd, DDMeanStdTanh]
CATEGORIES = ['Full', 'Wrist', 'Shoulder', 'Humerus']

SINGLE_MODELS = [Mura]
MULTI_MODELS = [FusionMean]
CATEGORIES = ['Humerus']
