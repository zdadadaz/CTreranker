import pymedtermino
pymedtermino.LANGUAGE = "en"
pymedtermino.REMOVE_SUPPRESSED_CONCEPTS = True
from pymedtermino.all import *
from pymedtermino import *
from pymedtermino.snomedct import *

def main():
    # a = SNOMEDCT.search("Gestational Trophoblastic Disease")
    a = SNOMEDCT.search("Trophoblastic Neoplasms")
    print(a)
    print(a[0])
    # print(a[0].parents)
    # print(a[0].children)
    print(a[0].terms)

if __name__ == '__main__':
    main()