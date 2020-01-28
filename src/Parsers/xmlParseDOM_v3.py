#!/usr/bin/env python
# -*- coding: utf-8 -*-


import xml.etree.ElementTree as ET
from os import walk, path
import re
from string import punctuation
from nameparser import HumanName
from unicodedata import normalize, category
from multiprocessing import Pool, cpu_count, Value
from optparse import OptionParser
from time import time, ctime, sleep
import sys
import datetime as dt
from zipfile import ZipFile

NUM_PROCS = cpu_count()
cont_error_curr = Value('i', 0)
cont_error_id = Value('i', 0)
cont_error_form = Value('i', 0)

get_institut = re.compile('-\s[\s\w]*')


def make_cli_parser():
    """Make the command line interface parser."""
    usage = "\n\n".join(["python %prog INPUT_CSV OUTPUT_CSV",
                        """
    ARGUMENTS:
    -n: Number of processor to launch (default: cpu processors)
    INPUT_CSV: an input CSV file
    OUTPUT_CSV: an output file that will contain the trees\
    """])
    cli_parser = OptionParser(usage)
    cli_parser.add_option('-n', '--numprocs', type='int', default=NUM_PROCS,
                          help="Number of processes to launch\
                          [DEFAULT: %default]")
    return cli_parser


start_time = time()


# 0  pesq_id
# 1  pesq_nome
# 2 pesq_nome_original
# 3  trab_instituicao
# 4  pesq_grau
# 5  stud_pd_nome
# 6  periodo_pd
# 7  ano_obtencao_pd
# 8 bolsa_nome_pd
# 9 pesq_subarea_pd
# 10 pesq_area_pd
# 11 pesq_grande_area_pd
# 12  pesq_dr_orientador_nome
# 13  pesq__dr_orientador_id
# 14  stud_dr_nome
# 15  pesq_tese
# 16  periodo_dr
# 17  ano_obtencao_dr
# 18 bolsa_nome_dr
# 19 pesq_subarea_dr
# 20 pesq_area_dr
# 21 pesq_grande_area_dr
# 22  pesq_ms_orientador_nome
# 23  pesq__ms_orientador_id
# 24  stud_ms_nome
# 25  pesq_diss
# 26  periodo_ms
# 27  ano_obtencao_ms
# 28 bolsa_nome_ms
# 29 pesq_subarea_ms
# 30 pesq_area_ms
# 31 pesq_grande_area_ms
# 32 alunos Orienta Doutorado
# 33 alunos Co-orienta Doutorado
# 34 alunos Orienta Mestrado
# 35 alunos Co-orienta Mestrado
def stripAccents(s):
    return ''.join(c for c in normalize('NFD', s) if category(c) != 'Mn')


def repairName(s):
    '''
    Padronizar os nomes, removendo titulações, removendo comentários entre
    parentesis, números e por último tornando o nome no padrão
    "nome sobrenome" (Algumas bases aparecem "sobrenome, nome" como padrão)
    '''
    remove_comments = re.compile(u'\(.*\)|\[.*\]')
    s = re.sub(remove_comments, '', s)
    if '/' in s and len(s):
        s = s.split('/')[0].strip()

    if '&' in s and len(s):
        s = s.split('&')[0].strip()

    if ' e ' in s and len(s):
        if len(s.split(' e ')[1].split()) > 1:
            s = s.split(' e ')[0].strip()

    if ';' in s and len(s):
        s = s.split(';')[0].strip()

    if 'co-or' in s and len((s.split('co-or')[0]).split()) > 1:
        s = s.split('co-or')[0].strip()

    if '-' in s and len((s.split('-')[0]).split()) > 2 and\
            len((s.split('-')[1]).split()) > 2:
        s = s.split('-')[0].strip()

    if ',' in s and len((s.split(',')[0]).split()) > 2 and\
            len((s.split(',')[1]).split()) > 2:
        s = s.split(',')[0].strip()

    s = s.lower() + '\n'
    if ' jr ' in s or ' jr\n' in s:
        s = s.replace(' jr ', ' junior ')
        s = s.replace(' jr\n', ' junior\n')

    remove_dr_prof = re.compile(u'(prof(essor|esseur)?(a)?\.?\s)|(d(oc(to)\
?)?(r)(a)?\.?\s)|(doc\.?\s)|(assist(ant|ente)?\s)|(assoc(iate)?\s)|(adjunto\s)|\
(ph\.?d\s)|(advisor\s)|(supervisor)|((co)?(-)?orientador)')
    remove_numbers = re.compile(u'[0-9]+')
    remove_punctuation_map = dict((ord(char), u' ') for char in punctuation)
    return (re.sub(remove_dr_prof, '', re.sub(remove_numbers, '',
                   str(HumanName(unicode(HumanName(stripAccents(s)
                       .translate(remove_punctuation_map).lower()
                       .strip())))))))


def getYear(data=''):
    match = re.search(r'\d{4}', data)
    year = int(match.group(0)) if match else 9999
    if year > 1800 and year < dt.datetime.now().year:
        return str(year)
    else:
        return 'None'


'''
Errors in Alunos Nomes:

Validar a rede com o o número de pesquisadores presentes no curriculo Lattes
como base e o grau dos nodos na rede.?????

'''


def buildFile(zip_file):
    pesq_id = 'None'
    pesq_nome = 'None'
    pesq_nome_orig = 'None'
    pesq_trab_nome = 'None'
    pesq_grau = 'None'
    pesq_posdoutorado_list = []
    pesq_doutorado_list = []
    pesq_mestrado_list = []
    alunos_or_mestrado = []
    alunos_or_doutorado = []
    alunos_coor_mestrado = []
    alunos_coor_doutorado = []
    with ZipFile(zip_file) as myzip:
        with myzip.open('curriculo.xml') as xml_file:
            curriculo = ET.parse(xml_file)
    tree_root = curriculo.getroot()
    if tree_root.find('DADOS-GERAIS') is not None:
        if 'NUMERO-IDENTIFICADOR' in tree_root.attrib and tree_root.attrib['NUMERO-IDENTIFICADOR']:
            pesq_id = tree_root.attrib['NUMERO-IDENTIFICADOR']\
                .strip().lower()
            if pesq_id == '9089204821424223':
                print 'Curriculo Laender: ', zip_file
        else:
            # print 'ERRO ID: ', zip_file
            cont_error_id.value += 1
        pesq_nome = repairName(unicode(tree_root.find('DADOS-GERAIS').attrib['NOME-COMPLETO'].strip().lower()))
        # pesq_nome_orig = unicode(tree_root.find('DADOS-GERAIS').attrib['NOME-COMPLETO'].strip())
        # return(pesq_id, '{0}\t{1}\t{2}\n'.format(zip_file, pesq_id, pesq_nome_orig.encode('utf-8')))

        if tree_root.find('DADOS-GERAIS').find('ENDERECO').find('ENDERECO-PROFISSIONAL') is not None:
            if 'NOME-INSTITUICAO-EMPRESA' in tree_root.find('DADOS-GERAIS').find('ENDERECO').find('ENDERECO-PROFISSIONAL').attrib:
                pesq_trab_nome = tree_root.find('DADOS-GERAIS').find('ENDERECO').find('ENDERECO-PROFISSIONAL').attrib['NOME-INSTITUICAO-EMPRESA']
        else:
            print 'ERRO ENDERECO: ', zip_file

        for formacao in set(tree_root.find('DADOS-GERAIS').find('FORMACAO-ACADEMICA-TITULACAO').findall('MESTRADO')) | set(tree_root.find('DADOS-GERAIS').find('FORMACAO-ACADEMICA-TITULACAO').findall('MESTRADO-PROFISSIONALIZANTE')):
            pesq_orientador_ms_nome = 'None'
            pesq_orientador_ms_id = 'None'
            pesq_stud_ms_nome = 'None'
            pesq_diss = 'None'
            periodo_ms = 'None'
            pesq_ano_obtencao_ms = 'None'
            pesq_bolsa_ms_nome = 'None'
            pesq_subarea_ms = 'None'
            pesq_area_ms = 'None'
            pesq_grande_area_ms = 'None'
            if 'STATUS-DO-CURSO' in formacao.attrib:
                if formacao.attrib['STATUS-DO-CURSO'] == 'CONCLUIDO':
                    pesq_grau = formacao.tag
                    pesq_orientador_ms_nome = repairName(unicode(formacao.attrib['NOME-COMPLETO-DO-ORIENTADOR'].strip().lower()))
                    if 'NUMERO-ID-ORIENTADOR' in formacao.attrib and formacao.attrib['NUMERO-ID-ORIENTADOR']:
                        pesq_orientador_ms_id = formacao.attrib['NUMERO-ID-ORIENTADOR']
                    pesq_diss = formacao.attrib['TITULO-DA-DISSERTACAO-TESE'].encode('utf-8')
                    pesq_stud_ms_nome = formacao.attrib['NOME-INSTITUICAO'].encode('utf-8')
                    periodo_ms = ' & '.join([formacao.attrib['ANO-DE-INICIO'].strip().lower(), formacao.attrib['ANO-DE-CONCLUSAO'].strip().lower()])
                    pesq_ano_obtencao_ms = formacao.attrib['ANO-DE-CONCLUSAO'].strip().lower()
                    if 'FLAG-BOLSA' in formacao.attrib:
                        if formacao.attrib['FLAG-BOLSA'] == 'SIM':
                            pesq_bolsa_ms_nome = formacao.attrib['NOME-AGENCIA'].encode('utf-8')
                        else:
                            pesq_bolsa_ms_nome = 'None'
                    if formacao.find('AREAS-DO-CONHECIMENTO') is not None:
                        for area_conhecimento in formacao.find('AREAS-DO-CONHECIMENTO'):
                            if 'NOME-DA-SUB-AREA-DO-CONHECIMENTO' in area_conhecimento.attrib:
                                if pesq_subarea_ms == 'None':
                                    pesq_subarea_ms = area_conhecimento.attrib['NOME-DA-SUB-AREA-DO-CONHECIMENTO'].strip().lower()
                                if area_conhecimento.attrib['NOME-DA-SUB-AREA-DO-CONHECIMENTO'] and area_conhecimento.attrib['NOME-DA-SUB-AREA-DO-CONHECIMENTO'].strip().lower() not in pesq_subarea_ms:
                                    pesq_subarea_ms = ' & '.join(set([pesq_subarea_ms, area_conhecimento.attrib['NOME-DA-SUB-AREA-DO-CONHECIMENTO'].strip().lower()]))
                            if 'NOME-DA-AREA-DO-CONHECIMENTO' in area_conhecimento.attrib:
                                if pesq_area_ms == 'None':
                                    pesq_area_ms = area_conhecimento.attrib['NOME-DA-AREA-DO-CONHECIMENTO'].strip().lower()
                                if area_conhecimento.attrib['NOME-DA-AREA-DO-CONHECIMENTO'] and area_conhecimento.attrib['NOME-DA-AREA-DO-CONHECIMENTO'].strip().lower() not in pesq_area_ms:
                                    pesq_area_ms = ' & '.join(set([pesq_area_ms, area_conhecimento.attrib['NOME-DA-AREA-DO-CONHECIMENTO'].strip().lower()]))
                            if 'NOME-GRANDE-AREA-DO-CONHECIMENTO' in area_conhecimento.attrib:
                                if pesq_grande_area_ms == 'None':
                                    pesq_grande_area_ms = area_conhecimento.attrib['NOME-GRANDE-AREA-DO-CONHECIMENTO'].strip().lower()
                                if area_conhecimento.attrib['NOME-GRANDE-AREA-DO-CONHECIMENTO'] and area_conhecimento.attrib['NOME-GRANDE-AREA-DO-CONHECIMENTO'].strip().lower() not in pesq_grande_area_ms:
                                    pesq_grande_area_ms = ' & '.join(set([pesq_grande_area_ms, area_conhecimento.attrib['NOME-GRANDE-AREA-DO-CONHECIMENTO'].strip().lower()]))
                    pesq_mestrado_list.append((pesq_orientador_ms_id.replace('\t', ' '), pesq_orientador_ms_nome.replace('\t', ' '), pesq_stud_ms_nome.replace('\t', ' ').replace('\n', ' '), pesq_diss.replace('\t', ' ').replace('\n', ' '), periodo_ms.replace('\t', ' '), pesq_ano_obtencao_ms.replace('\t', ' '), pesq_bolsa_ms_nome.replace('\t', ' ').replace('\n', ' '), pesq_subarea_ms.replace('\t', ' '), pesq_area_ms.replace('\t', ' '), pesq_grande_area_ms.replace('\t', ' ')))
            else:
                print 'ERRO STATUS MS: ', zip_file
        for formacao in tree_root.find('DADOS-GERAIS').find('FORMACAO-ACADEMICA-TITULACAO').findall('DOUTORADO'):
            pesq_orientador_dr_nome = 'None'
            pesq_orientador_dr_id = 'None'
            pesq_stud_dr_nome = 'None'
            pesq_tese = 'None'
            periodo_dr = 'None'
            pesq_ano_obtencao_dr = 'None'
            pesq_bolsa_dr_nome = 'None'
            pesq_subarea_dr = 'None'
            pesq_area_dr = 'None'
            pesq_grande_area_dr = 'None'
            if 'STATUS-DO-CURSO' in formacao.attrib:
                if formacao.attrib['STATUS-DO-CURSO'] == 'CONCLUIDO':
                    pesq_grau = formacao.tag
                    pesq_orientador_dr_nome = repairName(unicode(formacao.attrib['NOME-COMPLETO-DO-ORIENTADOR'].strip().lower()))
                    if 'NUMERO-ID-ORIENTADOR' in formacao.attrib and formacao.attrib['NUMERO-ID-ORIENTADOR']:
                        pesq_orientador_dr_id = formacao.attrib['NUMERO-ID-ORIENTADOR'].encode('utf-8')
                    pesq_tese = formacao.attrib['TITULO-DA-DISSERTACAO-TESE'].encode('utf-8')
                    pesq_stud_dr_nome = formacao.attrib['NOME-INSTITUICAO'].encode('utf-8')
                    periodo_dr = ' & '.join([formacao.attrib['ANO-DE-INICIO'].strip().lower(), formacao.attrib['ANO-DE-CONCLUSAO'].strip().lower()])
                    pesq_ano_obtencao_dr = formacao.attrib['ANO-DE-CONCLUSAO'].strip().lower()
                    if 'FLAG-BOLSA' in formacao.attrib:
                        if formacao.attrib['FLAG-BOLSA'] == 'SIM':
                            pesq_bolsa_dr_nome = formacao.attrib['NOME-AGENCIA'].encode('utf-8')
                        else:
                            pesq_bolsa_dr_nome = 'None'
                    if formacao.find('AREAS-DO-CONHECIMENTO') is not None:
                        for area_conhecimento in formacao.find('AREAS-DO-CONHECIMENTO'):
                            if pesq_subarea_dr == 'None':
                                pesq_subarea_dr = area_conhecimento.attrib['NOME-DA-SUB-AREA-DO-CONHECIMENTO'].strip().lower()
                            if area_conhecimento.attrib['NOME-DA-SUB-AREA-DO-CONHECIMENTO'] and area_conhecimento.attrib['NOME-DA-SUB-AREA-DO-CONHECIMENTO'].strip().lower() not in pesq_subarea_dr:
                                pesq_subarea_dr = ' & '.join(set([pesq_subarea_dr, area_conhecimento.attrib['NOME-DA-SUB-AREA-DO-CONHECIMENTO'].strip().lower()]))
                            if pesq_area_dr == 'None':
                                pesq_area_dr = area_conhecimento.attrib['NOME-DA-AREA-DO-CONHECIMENTO'].strip().lower()
                            if area_conhecimento.attrib['NOME-DA-AREA-DO-CONHECIMENTO'] and area_conhecimento.attrib['NOME-DA-AREA-DO-CONHECIMENTO'].strip().lower() not in pesq_area_dr:
                                pesq_area_dr = ' & '.join(set([pesq_area_dr, area_conhecimento.attrib['NOME-DA-AREA-DO-CONHECIMENTO'].strip().lower()]))
                            if pesq_grande_area_dr == 'None':
                                pesq_grande_area_dr = area_conhecimento.attrib['NOME-GRANDE-AREA-DO-CONHECIMENTO'].strip().lower()
                            if area_conhecimento.attrib['NOME-GRANDE-AREA-DO-CONHECIMENTO'] and area_conhecimento.attrib['NOME-GRANDE-AREA-DO-CONHECIMENTO'].strip().lower() not in pesq_grande_area_dr:
                                pesq_grande_area_dr = ' & '.join(set([pesq_grande_area_dr, area_conhecimento.attrib['NOME-GRANDE-AREA-DO-CONHECIMENTO'].strip().lower()]))
                    pesq_doutorado_list.append((pesq_orientador_dr_id.replace('\t', ' '), pesq_orientador_dr_nome.replace('\t', ' '), pesq_stud_dr_nome.replace('\t', ' ').replace('\n', ' '), pesq_tese.replace('\t', ' ').replace('\n', ' '), periodo_dr.replace('\t', ' '), pesq_ano_obtencao_dr.replace('\t', ' '), pesq_bolsa_dr_nome.replace('\t', ' ').replace('\n', ' '), pesq_subarea_dr.replace('\t', ' '), pesq_area_dr.replace('\t', ' '), pesq_grande_area_dr.replace('\t', ' ')))
            else:
                print 'ERRO STATUS DR: ', zip_file
        for formacao in tree_root.find('DADOS-GERAIS').find('FORMACAO-ACADEMICA-TITULACAO').findall('POS-DOUTORADO'):
            pesq_stud_pd_nome = 'None'
            periodo_pd = 'None'
            ano_obtencao_pd = 'None'
            bolsa_nome_pd = 'None'
            pesq_subarea_pd = 'None'
            pesq_area_pd = 'None'
            pesq_grande_area_pd = 'None'
            if 'STATUS-DO-CURSO' in formacao.attrib:
                if formacao.attrib['STATUS-DO-CURSO'] == 'CONCLUIDO':
                    pesq_grau = formacao.tag
                    pesq_stud_pd_nome = formacao.attrib['NOME-INSTITUICAO'].encode('utf-8')
                    periodo_pd = ' & '.join([formacao.attrib['ANO-DE-INICIO'].strip().lower(), formacao.attrib['ANO-DE-CONCLUSAO'].strip().lower()])
                    ano_obtencao_pd = formacao.attrib['ANO-DE-CONCLUSAO'].strip().lower()
                    if 'FLAG-BOLSA' in formacao.attrib:
                        if formacao.attrib['FLAG-BOLSA'] == 'SIM':
                            bolsa_nome_pd = formacao.attrib['NOME-AGENCIA'].encode('utf-8')
                    if formacao.find('AREAS-DO-CONHECIMENTO') is not None:
                        for area_conhecimento in formacao.find('AREAS-DO-CONHECIMENTO'):
                            if pesq_subarea_pd == 'None':
                                pesq_subarea_pd = area_conhecimento.attrib['NOME-DA-SUB-AREA-DO-CONHECIMENTO'].strip().lower()
                            if area_conhecimento.attrib['NOME-DA-SUB-AREA-DO-CONHECIMENTO'] and area_conhecimento.attrib['NOME-DA-SUB-AREA-DO-CONHECIMENTO'].strip().lower() not in pesq_subarea_pd:
                                pesq_subarea_pd = ' & '.join(set([pesq_subarea_pd, area_conhecimento.attrib['NOME-DA-SUB-AREA-DO-CONHECIMENTO'].strip().lower()]))
                            if pesq_area_pd == 'None':
                                pesq_area_pd = area_conhecimento.attrib['NOME-DA-AREA-DO-CONHECIMENTO'].strip().lower()
                            if area_conhecimento.attrib['NOME-DA-AREA-DO-CONHECIMENTO'] and area_conhecimento.attrib['NOME-DA-AREA-DO-CONHECIMENTO'].strip().lower() not in pesq_area_pd:
                                pesq_area_pd = ' & '.join(set([pesq_area_pd, area_conhecimento.attrib['NOME-DA-AREA-DO-CONHECIMENTO'].strip().lower()]))
                            if pesq_grande_area_pd == 'None':
                                pesq_grande_area_pd = area_conhecimento.attrib['NOME-GRANDE-AREA-DO-CONHECIMENTO'].strip().lower()
                            if area_conhecimento.attrib['NOME-GRANDE-AREA-DO-CONHECIMENTO'] and area_conhecimento.attrib['NOME-GRANDE-AREA-DO-CONHECIMENTO'].strip().lower() not in pesq_grande_area_pd:
                                pesq_grande_area_pd = ' & '.join(set([pesq_grande_area_pd, area_conhecimento.attrib['NOME-GRANDE-AREA-DO-CONHECIMENTO'].strip().lower()]))
                    pesq_posdoutorado_list.append((pesq_stud_pd_nome.replace('\t', ' ').replace('\n', ' '), periodo_pd.replace('\t', ' '), ano_obtencao_pd.replace('\t', ' '), bolsa_nome_pd.replace('\t', ' ').replace('\n', ' '), pesq_subarea_pd.replace('\t', ' '), pesq_area_pd.replace('\t', ' '), pesq_grande_area_pd.replace('\t', ' ')))
            else:
                print 'ERRO STATUS PD: ', zip_file
        if tree_root.find('OUTRA-PRODUCAO') is not None:
            if tree_root.find('OUTRA-PRODUCAO').find('ORIENTACOES-CONCLUIDAS') is not None:
                for orientacao in tree_root.find('OUTRA-PRODUCAO').find('ORIENTACOES-CONCLUIDAS').findall('ORIENTACOES-CONCLUIDAS-PARA-MESTRADO'):
                    id_aluno_lattes = 'None'
                    nome_aluno = 'None'
                    titulo_aluno = 'None'
                    ano_aluno = 'None'
                    role = 'None'
                    inst_aluno = 'None'
                    if orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-MESTRADO') is not None:
                        if 'NUMERO-ID-ORIENTADO' in orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-MESTRADO').attrib and orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-MESTRADO').attrib['NUMERO-ID-ORIENTADO']:
                            id_aluno_lattes = orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-MESTRADO').attrib['NUMERO-ID-ORIENTADO'].strip().lower()
                        if 'NOME-DO-ORIENTADO' in orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-MESTRADO').attrib:
                            nome_aluno = repairName(unicode(orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-MESTRADO').attrib['NOME-DO-ORIENTADO'].strip().lower()))
                        if 'TIPO-DE-ORIENTACAO' in orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-MESTRADO').attrib:
                            role = orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-MESTRADO').attrib['TIPO-DE-ORIENTACAO'].strip().lower()
                        if 'NOME-DA-INSTITUICAO' in orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-MESTRADO').attrib:
                            inst_aluno = orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-MESTRADO').attrib['NOME-DA-INSTITUICAO']
                        if orientacao.find('DADOS-BASICOS-DE-ORIENTACOES-CONCLUIDAS-PARA-MESTRADO') is not None:
                            if 'TITULO' in orientacao.find('DADOS-BASICOS-DE-ORIENTACOES-CONCLUIDAS-PARA-MESTRADO').attrib:
                                titulo_aluno = orientacao.find('DADOS-BASICOS-DE-ORIENTACOES-CONCLUIDAS-PARA-MESTRADO').attrib['TITULO']
                            if 'ANO' in orientacao.find('DADOS-BASICOS-DE-ORIENTACOES-CONCLUIDAS-PARA-MESTRADO').attrib:
                                if orientacao.find('DADOS-BASICOS-DE-ORIENTACOES-CONCLUIDAS-PARA-MESTRADO').attrib['ANO']:
                                    ano_aluno = orientacao.find('DADOS-BASICOS-DE-ORIENTACOES-CONCLUIDAS-PARA-MESTRADO').attrib['ANO'].strip().lower()
                        if role == 'orientador_principal':
                            alunos_or_mestrado.append((id_aluno_lattes, nome_aluno.replace('\t', ' '),
                                                       titulo_aluno.encode('utf-8').replace('\t', ' ').replace('\n', ' '), ano_aluno.replace('\t', ' '),
                                                       inst_aluno.encode('utf-8').replace('\t', ' ').replace('\n', ' ')))
                        elif role == 'co_orientador':
                            alunos_coor_mestrado.append((id_aluno_lattes,
                                                         nome_aluno.replace('\t', ' '), titulo_aluno.encode('utf-8').replace('\t', ' ').replace('\n', ' '),
                                                         ano_aluno.replace('\t', ' '), inst_aluno.encode('utf-8').replace('\t', ' ').replace('\n', ' ')))
                for orientacao in tree_root.find('OUTRA-PRODUCAO').find('ORIENTACOES-CONCLUIDAS').findall('ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO'):
                    id_aluno_lattes = 'None'
                    nome_aluno = 'None'
                    titulo_aluno = 'None'
                    ano_aluno = 'None'
                    role = 'None'
                    inst_aluno = 'None'
                    if orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO') is not None:
                        if 'NUMERO-ID-ORIENTADO' in orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO').attrib and orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO').attrib['NUMERO-ID-ORIENTADO']:
                            id_aluno_lattes = orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO').attrib['NUMERO-ID-ORIENTADO'].strip().lower()
                        if 'NOME-DO-ORIENTADO' in orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO').attrib:
                            nome_aluno = repairName(unicode(orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO').attrib['NOME-DO-ORIENTADO'].strip().lower()))
                        if 'TIPO-DE-ORIENTACAO' in orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO').attrib:
                            role = orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO').attrib['TIPO-DE-ORIENTACAO'].strip().lower()
                        if 'NOME-DA-INSTITUICAO' in orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO').attrib:
                            inst_aluno = orientacao.find('DETALHAMENTO-DE-ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO').attrib['NOME-DA-INSTITUICAO']
                        if orientacao.find('DADOS-BASICOS-DE-ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO') is not None:
                            if 'TITULO' in orientacao.find('DADOS-BASICOS-DE-ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO').attrib:
                                titulo_aluno = orientacao.find('DADOS-BASICOS-DE-ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO').attrib['TITULO']
                            if 'ANO' in orientacao.find('DADOS-BASICOS-DE-ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO').attrib:
                                if orientacao.find('DADOS-BASICOS-DE-ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO').attrib['ANO']:
                                    ano_aluno = orientacao.find('DADOS-BASICOS-DE-ORIENTACOES-CONCLUIDAS-PARA-DOUTORADO').attrib['ANO'].strip().lower()
                        if role == 'orientador_principal':
                            alunos_or_doutorado.append((id_aluno_lattes, nome_aluno.replace('\t', ' '),
                                                       titulo_aluno.encode('utf-8').replace('\t', ' ').replace('\n', ' '), ano_aluno.replace('\t', ' '),
                                                       inst_aluno.encode('utf-8').replace('\t', ' ').replace('\n', ' ')))
                        elif role == 'co_orientador':
                            alunos_coor_doutorado.append((id_aluno_lattes,
                                                         nome_aluno.replace('\t', ' '), titulo_aluno.encode('utf-8').replace('\t', ' ').replace('\n', ' '),
                                                         ano_aluno.replace('\t', ' '), inst_aluno.encode('utf-8').replace('\t', ' ').replace('\n', ' ')))
    else:
        print 'ERRO CURRICULO: ', zip_file
        cont_error_curr.value += 1
        return ('ERROR', '')
    if not pesq_doutorado_list:
        cont_error_form.value += 1
        return ('ERROR', '')

    return(pesq_id, '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\
\t{11}\t{12}\n'
           .format(str(pesq_id.replace('\t', ' '))
                   if str(pesq_id.replace('\t', ' ')) else 'None',
                   pesq_nome.replace('\t', ' ')
                   if pesq_nome.replace('\t', ' ') else 'None',
                   pesq_nome_orig.replace('\t', ' ').encode('utf-8')
                   if pesq_nome_orig.replace('\t', ' ') else 'None',
                   pesq_trab_nome.replace('\t', ' ').encode('utf-8').replace('\n', ' ')
                   if pesq_trab_nome.replace('\t', ' ') else 'None',
                   pesq_grau.replace('\t', ' ').encode('utf-8')
                   if pesq_grau.replace('\t', ' ') else 'None',
                   str(pesq_posdoutorado_list).encode('utf-8'),
                   str(pesq_doutorado_list).encode('utf-8'),
                   str(pesq_mestrado_list).encode('utf-8'),
                   str(alunos_or_doutorado).encode('utf-8'),
                   str(alunos_coor_doutorado).encode('utf-8'),
                   str(alunos_or_mestrado).encode('utf-8'),
                   str(alunos_coor_mestrado).encode('utf-8'),
                   pesq_ano_obtencao_dr.encode('utf-8')))


def main(argv):
    cli_parser = make_cli_parser()
    opts, args = cli_parser.parse_args(argv)
    numprocs = opts.numprocs
    set_curr_ids = set()
    print 'number of processes: ', numprocs
    xml_files = []
    for (dirpath, dirnames, filenames) in walk('/home/murai/Projects/data/datasets/lattes/collection/00'):
        xml_files.extend(path.join(dirpath, filename) for filename in
                         filenames)
    pool = Pool(processes=numprocs)
# /home/wellington/Bases de Dados/curriculos/curriculos_2
# /178360_3457219624656691.xml
    print(len(xml_files))
    print('Writing File')
    debugger = True
    cont = 0
    # if debugger:
    with open('lattes_v11.csv', 'w') as f:
        f.write('pesq_id\tpesq_nome\tpesq_nome_orig\tinst_pertence_nome\tpesq_grau\
\tposdoutorado_list\t\doutorado_list\tmestrado_list\talunos_or_doutorado\talunos_coor_doutorado\
\talunos_or_mestrado\talunos_coor_mestrado\tano_obtencao_dr\n')
        # set_curr_ids.add('ERROR')
        # buildFile('/home/wellington/Bases de Dados/Base Lattes/collection/58/75288343819683.zip')
        # exit()
        # for xml in xml_files:
        #     buildFile(xml)
        #     if cont % 1000 == 0:
        #         print cont, ctime()
        #     cont += 1
        #     sleep(5)
        for curr_id, file_str in pool.imap_unordered(buildFile, xml_files):
            # if curr_id not in set_curr_ids:
            cont += 1
            if file_str:
                # set_curr_ids.add(curr_id)
                f.write(file_str)
            if cont % 1000 == 0:
                print cont, ctime()
            # else:
            #     print file_str


if __name__ == '__main__':
    start_time = time()
    print 'Process started time: ', str(ctime())
    main(sys.argv[1:])
    print('Erros IDs: ', cont_error_id.value)
    print('Erros Curr: ', cont_error_curr.value)
    print('Erros Form: ', cont_error_form.value)

    print('End of Program! Time: %s' % (time() - start_time))



'''

('Erros IDs: ', 2753)
('Erros Curr: ', 434)
End of Program! Time: 3655.49338603


Nova Base versao XML oficial Lattes (mil vezes melhor pena que chegou tarde :/ :(
# ##Informacaoo Pessoal###
# 0  pesq_id
# 1 data_atualizacao
# 2  pesq_nome
# 3 pesq_nome_original
# 4 nome em citação
# 5 nacionalidade
# 6 pais nascimento
# 7 cidade nascimento
# 8 sigla pais nascimento
# 9 pais nacionalidade
# 10 codigo instituicao
# 11 nome instituicao
# 12 sigla instituicao
# 12 codigo orgao
# 13 nome orgao
# 14 codigo unidade
# 15 nome unidade
# 16 pais instituicao
# 17 cep instituicao
# 18 estado instituicao
# 19 cidade instituicao
# 20 homepage
# ##Formacao Academica Mestrado###
# 21 nivel Formacao ms
# 22 codigo instituicao ms
# 23 nome instituicao ms
# 24 pais instituicao ms
# 25 sigla pais instituicao ms
# 25 sigla instituicao ms
# 24 codigo orgao ms
# 25 nome orgao ms
# 26 codigo do curso ms
# 27 nome do curso ms
# 28 codigo area do curso ms
# 29 status do curso ms
# 30 ano inicio ms
# 31 ano conclusao ms
# 32 flag recebeu bolsa ms
# 33 codigo agencia bolsa ms
# 34 nome agencia bolsa ms
# 35 ano obtencao titulo ms
# 36 titulo dissertacao ms
# 37 nome orientador ms
# 38 id orientador ms
# 39 tipo ms
# 40 codigo curso capes ms
# 41 titulo dissertacao ingles ms
# 42 nome curso ingles ms
# 43 nome co orientador ms
# 44 codigo instituicao dout ms
# 45 nome instituicao dout ms
# 46 codigo instituicao outra dout ms
# 47 nome instituicao outra dout ms
# 48 nome orientador dout ms
# 49 lista palavras chave ms
# 50 lista areas do conhecimento
# ##Formacao Academica Doutorado###
# 51 nivel Formacao dr
# 52 codigo instituicao dr
# 53 nome instituicao dr
# 24 pais instituicao dr
# 25 sigla pais instituicao dr
# 25 sigla instituicao dr
# 54 codigo orgao dr
# 55 nome orgao dr
# 56 codigo do curso dr
# 57 nome do curso dr
# 58 codigo area do curso dr
# 59 status do curso dr
# 60 ano inicio dr
# 61 ano conclusao dr
# 62 flag recebeu bolsa dr
# 63 codigo agencia bolsa dr
# 64 nome agencia bolsa dr
# 65 ano obtencao titulo dr
# 66 titulo tese dr
# 67 nome orientador dr
# 68 id orientador dr
# 69 tipo dr
# 70 codigo curso capes dr
# 71 titulo dissertacao ingles dr
# 72 nome curso ingles dr
# 73 nome co orientador dr
# 74 codigo instituicao dout dr
# 75 nome instituicao dout dr
# 76 codigo instituicao outra dout dr
# 77 nome instituicao outra dout dr
# 78 nome orientador dout dr
# 79 nome orientador co tut dr
# 80 codigo instituicao outra co tut dr
# 81 codigo instituicao co tut dr
# 82 nome orientador sand dr
# 83 codigo instituicao outra sand dr
# 84 codigo instituicao sand dr
# 85 lista palavras chave dr
# 86 lista area do conhecimento dr
# ##Formacao Pos Doutorado ##
# 87 nivel pos dr
# 88 codigo instituicao posdr
# 89 nome instituicao posdr
# 90 ano inicio posdr
# 91 ano conclusao posdr
# 92 ano obtencao posdr
# 93 flag bolsa posdr
# 94 codigo agencia posdr
# 95 nome agencia posdr
# 96 status estagio posdr
# 97 status curso posdr
# 98 numero id orientador posdr
# 99 codigo curso capes posdr
# 100 titulo do trabalho posdr
# 101 titulo trabalho ingles posdr
# 102 nome curso ingles posdr
# 103 lista palavras chave posdr
# 104 lista area do conhecimento posdr
# ## areas atuacao pesquisador ##
# 105 lista areas de atuacao
'''
