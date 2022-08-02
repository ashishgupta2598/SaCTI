
from logging import exception


def get_path(exp_type):
    if exp_type=='saCTI-large coarse':
        train = '../sacti/coarse/train.conll'
        dev = '../sacti/coarse/dev.conll'
        test = '../sacti/coarse/test.conll'
    elif exp_type=='saCTI-large fine':
        train = '../sacti/fine/train.conll'
        dev = '../sacti/fine/dev.conll'
        test = '../sacti/fine/test.conll'
    elif exp_type=='saCTI-base coarse':
        train = '../coling/coarse/train.conll'
        dev = '../coling/coarse/dev.conll'
        test = '../coling/coarse/test.conll'
    elif exp_type=='saCTI-base fine':
        train = '../coling/fine/train.conll'
        dev = '../coling/fine/dev.conll'
        test = '../coling/fine/test.conll'
    elif exp_type=='marathi':
        train = '../marathi/coarse/train.conll'
        dev = '../marathi/coarse/dev.conll'
        test = '../marathi/coarse/test.conll'
    elif exp_type=='english':
        train = '../english/coarse/train.conll'
        dev = '../english/coarse/dev.conll'
        test = '../english/coarse/test.conll'
    else:
        raise Exception("Please select a proper experimnet from the list")

    return train,dev,test

choices = ['saCTI-large coarse','saCTI-large fine','saCTI-base coarse','saCTI-base fine','marathi','english']