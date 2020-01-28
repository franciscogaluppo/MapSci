import xml.etree.ElementTree as ET
from zipfile import ZipFile
from os import listdir

# PESQUISADOR
# 0 id_pesq
# 1 nome_instituicao
# 2 cep_instituicao

# PUBLICAÇÃO
# 0 id_pesq
# 1 n_pesq
# 2 nome_pub
# 3 ano_pub
# 4 doi_pub
# 5 journal_pub

def get_root(zip_file):
    with ZipFile(zip_file) as myzip:
        with myzip.open('curriculo.xml') as xml_file:
            curriculo = ET.parse(xml_file)
    return curriculo.getroot()

def get_id(root):
    if 'NUMERO-IDENTIFICADOR' in root.attrib and root.attrib['NUMERO-IDENTIFICADOR']:
        return root.attrib['NUMERO-IDENTIFICADOR'].strip().lower()
    print('ERRO ID: ', zip_file)

def get_nome_instituicao(root):
    end = root.find('DADOS-GERAIS/ENDERECO/ENDERECO-PROFISSIONAL')
    if end is not None:
        if 'NOME-INSTITUICAO-EMPRESA' in end.attrib:
            return end.attrib['NOME-INSTITUICAO-EMPRESA']

def get_cep_instituicao(root):
    end = root.find('DADOS-GERAIS/ENDERECO/ENDERECO-PROFISSIONAL')
    if end is not None:
        if 'CEP' in end.attrib:
            return end.attrib['CEP']

def parse_file(zip_file, arq_pesq, arq_pub):
    """
    Dado um arquivo de leitura (zip_file), extrai as informações
    que precisamos para os arquivos de saída (arq_pesq e arq_pub)
    """
    id_pesq = 'None'
    nome_instituicao = 'None'
    cep_instituicao = 'None'

    tree_root = get_root(zip_file)
    
    if tree_root.find('DADOS-GERAIS') is not None:
        id_pesq = get_id(tree_root)
        nome_instituicao = get_nome_instituicao(tree_root)
        cep_instituicao = get_cep_instituicao(tree_root)

    arq_pesq.write("{}, {}, {}\n".format(id_pesq, nome_instituicao, cep_instituicao))
    
    artigos = tree_root.find("PRODUCAO-BIBLIOGRAFICA/ARTIGOS-PUBLICADOS")
    if artigos is not None:
        for pub in artigos:
        
            #nome_pub = None
            #ano_pub = None
            #doi_pub = None
            #journal_pub = None
            #n_pesq = None

            # ADICIONAR OS IFS
        
            nome_pub = pub[0].attrib["TITULO-DO-ARTIGO"] 
            ano_pub = pub[0].attrib["ANO-DO-ARTIGO"]
            doi_pub = pub[0].attrib["DOI"]
            journal_pub = pub[1].attrib["TITULO-DO-PERIODICO-OU-REVISTA"]
            n_pesq = len([x for x in pub.iter("AUTORES")])
        
            arq_pub.write("{}, {}, {}, {}, {}, {}\n".format(
                id_pesq, nome_pub, ano_pub, doi_pub, journal_pub, n_pesq
                ))


if __name__ == '__main__':
    print("Processo iniciado")
    with open("pesquisadores.csv", "w") as arq_pesq:
        with open("artigos.csv", "w") as arq_pub:
            arq_pesq.write("id_pesquisador, nome_instituicao, cep_instituicao\n")
            arq_pub.write("id_pesquisador, nome_artigo, ano_publicacao, DOI, journal_ou_conferencia, numero_pesquisadores\n")
            for diretorio in range(100):
                path = "/data/" + str(diretorio).zfill(2)
                for filez in listdir(path):
                    zip_file = path + filez
                    parse_file(zip_file, arq_pesq, arq_pub)
