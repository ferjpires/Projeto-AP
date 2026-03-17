import itertools

def build_grid(dictionary):
    """
    Função baseada no guião da aula_05_03.
    Recebe um dicionário com os hiperparâmetros a testar e devolve
    um gerador com todas as combinações possíveis.
    """
    keys = dictionary.keys()
    values = dictionary.values()
    for instance in itertools.product(*values):
        yield dict(zip(keys, instance))