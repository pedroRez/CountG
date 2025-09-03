# Contribuindo com CountG

Obrigado por considerar uma contribuição para o projeto! Este documento descreve o fluxo de trabalho adotado, o padrão de mensagens de commit e como preparar o ambiente de desenvolvimento.

## Fluxo de trabalho

1. Faça um fork do repositório e clone o seu fork.
2. Crie um branch descritivo para cada alteração (`git checkout -b minha-feature`).
3. Realize alterações pequenas e bem testadas.
4. Garanta que os testes e as ferramentas de lint passam.
5. Abra um Pull Request para a branch principal descrevendo claramente as mudanças.

## Padrão de commits

Utilizamos o formato [Conventional Commits](https://www.conventionalcommits.org/):

- `feat`: nova funcionalidade
- `fix`: correção de bug
- `docs`: alterações apenas na documentação
- `refactor`, `test`, `chore`, etc.

O formato básico é:

```
tipo(escopo opcional): descrição curta

Corpo (opcional)
```

Exemplo:

```
feat: adiciona rota para upload de vídeo
```

## Ambiente de desenvolvimento

1. Requer **Python 3.10+**.
2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Instale as dependências principais:
   ```bash
   pip install -r requirements.txt
   ```
4. Instale as dependências de desenvolvimento (lint e testes):
   ```bash
   pip install black flake8 pytest
   ```
5. Execute os testes e verificações de estilo antes de enviar o PR:
   ```bash
   flake8
   pytest
   ```

## Dúvidas

Em caso de dúvidas, abra uma issue explicando o problema ou entre em contato através das discussões do repositório.

Obrigado por contribuir!
