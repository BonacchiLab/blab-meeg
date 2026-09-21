# %%
import mne
import pandas as pd

subject = "CA124"
method = "mag"


epochs = mne.read_epochs(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Preproc\04_epochs\Phase1_onset_-100_500ms\{subject}_04_epochs_{method}_Phase1_epo.fif"
)

meta = epochs.metadata

# Opcional: ver os nomes das colunas disponíveis
print("Colunas disponíveis:", meta.columns.tolist())

# Opção 1: Normalizar para maiúsculas (se for só diferença de caixa)
meta["relevance"] = meta["relevance"].str.capitalize()
meta["category"] = meta["category"].str.capitalize()

# Agora sim, usar os nomes com maiúscula
tabela = pd.crosstab(meta["relevance"], meta["category"])

# Definir a ordem desejada das linhas (Target, Relevant, Irrelevant)
linhas_ordem = ["target", "relevant", "irrelevant"]
# Se alguma linha não existir, adiciona com zeros
tabela = tabela.reindex(linhas_ordem, fill_value=0)

# Definir a ordem desejada das colunas (Face, Object, Letter, False Font)
colunas_ordem = ["face", "object", "letter", "false Font"]
# Se alguma coluna não existir, adiciona com zeros
tabela = tabela.reindex(columns=colunas_ordem, fill_value=0)

# Imprimir a tabela formatada
print("\nTabela de contagens:")
print(tabela)

# (Opcional) guardar em ficheiro CSV
# tabela.to_csv('tabela_contagens.csv')

# %%
print("Valores únicos em 'relevance':", meta["relevance"].unique())
# %%
print("Número de trials:", len(meta))
# %%
import mne
import pandas as pd

subject = "CA124"
method = "mag"
epochs = mne.read_epochs(fr"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Preproc\04_epochs\Phase1_onset_-100_500ms\{subject}_04_epochs_{method}_Phase1_epo.fif")

meta = epochs.metadata
if meta is None:
    print("Sem metadata!")
else:
    # 1. Ver valores reais
    print("Categorias:", meta['category'].unique())
    print("Relevâncias:", meta['relevance'].unique())

    # 2. Normalize ou mapeie conforme necessário (exemplo com capitalização)
    meta['relevance'] = meta['relevance'].str.capitalize()
    meta['category'] = meta['category'].str.capitalize()

    # 3. Tabela cruzada
    tabela = pd.crosstab(meta['relevance'], meta['category'])

    # 4. Ordenar linhas e colunas (use os nomes exatos que agora têm)
    linhas_ordem = ['Target', 'Relevant', 'Irrelevant']
    colunas_ordem = ['Face', 'Object', 'Letter', 'False Font']
    tabela = tabela.reindex(index=linhas_ordem, columns=colunas_ordem, fill_value=0)

    # 5. Mostrar
    print(tabela)
# %%
import mne
import pandas as pd

subject = "CA124"
method = "mag"
epochs = mne.read_epochs(fr"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject}\Preproc\04_epochs\Phase1_onset_-100_500ms\{subject}_04_epochs_{method}_Phase1_epo.fif")

meta = epochs.metadata

if meta is None:
    print("Sem metadata!")
else:
    # 1. Remover linhas com relevance = None
    meta = meta[meta['relevance'].notna()].copy()

    # 2. Mapear os valores para os nomes desejados
    mapeamento_category = {
        'faces': 'Face',
        'objects': 'Object',
        'fonts': 'Letter',        # Ajuste se for outra categoria
        'false_fonts': 'False Font'
    }
    mapeamento_relevance = {
        'target': 'Target',
        'relevant': 'Relevant',
        'irrelevant': 'Irrelevant'
    }

    meta['category'] = meta['category'].map(mapeamento_category)
    meta['relevance'] = meta['relevance'].map(mapeamento_relevance)

    # 3. Remover eventuais linhas que não tenham correspondência (opcional)
    meta = meta.dropna(subset=['category', 'relevance'])

    # 4. Tabela cruzada
    tabela = pd.crosstab(meta['relevance'], meta['category'])

    # 5. Garantir a ordem das linhas e colunas
    linhas_ordem = ['Target', 'Relevant', 'Irrelevant']
    colunas_ordem = ['Face', 'Object', 'Letter', 'False Font']
    tabela = tabela.reindex(index=linhas_ordem, columns=colunas_ordem, fill_value=0)

    # 6. Exibir a tabela
    print(tabela)
# %%
