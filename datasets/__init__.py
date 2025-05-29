from datasets.colored_mnist import CMNIST
from datasets.colored_fmnist import CFMNIST
from datasets.colored_object import COCOcolor_LYPD
from datasets.pacs import PACS_FROM_DOMAINBED
from datasets.vlcs import VLCS_FROM_DOMAINBED
from datasets.domainnet import DomainNet_FROM_DOMAINBED
from datasets.terraIncognita import TerraIncognita_FROM_DOMAINBED
from datasets.camelyon import Camelyon_FROM_DOMAINBED
__all__ = ["CMNIST", 
           "CFMNIST", 
           "COCOcolor_LYPD", 
           "PACS_FROM_DOMAINBED", 
           "VLCS_FROM_DOMAINBED", 
           "DomainNet_FROM_DOMAINBED", 
           "TerraIncognita_FROM_DOMAINBED",
           "Camelyon_FROM_DOMAINBED"]
