# Configuração do Ambiente de Desenvolvimento com Python 3.9.5

## Sumário
1. [Instalação do Python 3.9.5](#instalação-do-python-395)
2. [Método 1: Ambiente Virtual com venv](#método-1-ambiente-virtual-com-venv)
3. [Método 2: Ambiente Virtual com Conda](#método-2-ambiente-virtual-com-conda)
4. [Comparação entre venv e Conda](#comparação-entre-venv-e-conda)
5. [Recomendações Finais](#recomendações-finais)

---

## Instalação do Python 3.9.5

### Windows
1. Acesse [python.org/downloads](https://www.python.org/downloads/)
2. Baixe o instalador para Windows (versão 3.9.5)
3. Execute o instalador marcando:
   - ☑ `Add Python 3.9 to PATH`
   - ☑ `Install launcher for all users`

**Verificação:**
```bash
python --version
# Deve retornar: Python 3.9.5
```

## Método 1: Ambiente Virtual com venv

### Passo-a-passo
1. Criar diretório do projeto:
   ```bash
   mkdir curso_cv
   cd curso_cv
   ```

2. Criar ambiente virtual:
   ```bash
   python -m venv curso_cv
   ```

3. Ativar ambiente:
   ```bash
   # Windows
   curso_cv\Scripts\activate

   # Linux/MacOS
   source curso_cv/bin/activate
   ```

4. Instalar pacotes:
   ```bash
   pip install opencv-python numpy matplotlib seaborn scikit-learn tensorflow
   ```

5. Verificar instalação:
   ```bash
   python -c "import cv2; print(cv2.__version__)"
   ```

6. Desativar ambiente:
   ```bash
   deactivate
   ```

---

## Método 2: Ambiente Virtual com Conda

### Pré-requisitos
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) instalado

### Passo-a-passo
1. Criar ambiente:
   ```bash
   conda create -n curso_cv python=3.9.5
   ```

2. Ativar ambiente:
   ```bash
   conda activate curso_cv
   ```

3. Instalar pacotes:
   ```bash
   conda install opencv-python numpy matplotlib seaborn scikit-learn tensorflow
   # ou
   pip install opencv-python numpy matplotlib seaborn scikit-learn tensorflow
   ```

4. Verificar instalação:
   ```bash
   python -c "import cv2; print(cv2.__version__)"
   ```

5. Desativar ambiente:
   ```bash
   conda deactivate
   ```

---

## Comparação entre venv e Conda

| Característica       | venv                          | Conda                         |
|----------------------|-------------------------------|-------------------------------|
| Origem               | Nativo do Python              | Anaconda/Miniconda            |
| Instalação           | `python -m venv`              | `conda create`                |
| Dependências complexas | Limitado                   | Melhor suporte                |
| Pacotes não-Python   | Não suportado                | Suportado (ex: CUDA, MKL)     |
| Performance          | Mais leve                    | Mais pesado                   |

---

## Recomendações Finais

✅ **Use venv quando:**
- Seu projeto é exclusivamente Python
- Precisa de uma solução leve e nativa

✅ **Prefira Conda quando:**
- Trabalha com Data Science/ML pesado
- Precisa de dependências não-Python (CUDA, etc)
- Usa frequentemente Jupyter Notebooks

> **Dica profissional:** Para projetos TensorFlow com GPU, o Conda geralmente oferece melhor compatibilidade com drivers CUDA.
