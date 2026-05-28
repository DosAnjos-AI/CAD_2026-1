# Instruções para Agentes de IA

## Fluxo de Trabalho Obrigatório

1. Analisar arquivos relevantes
2. Propor plano com riscos e dependências
3. Aguardar aprovação explícita
4. Implementar
5. Testar e validar
6. Commitar apenas após validação

Nunca implementar sem aprovação explícita do responsável.

## Código

- Proibido código morto — qualquer substituição de função, classe ou objeto
  exige revisão e remoção do que foi substituído
- Comentários em português
- Sem overengineering — soluções simples e diretas

## Git

- Fluxo: feature/xxx -> main
- Nunca commitar diretamente na main
- Commits em português seguindo Conventional Commits:
  feat:, fix:, chore:, docs:, refactor:, test:

## Ao Propor um Plano

- Listar arquivos que serão criados ou modificados
- Listar riscos conhecidos
- Listar dependências entre etapas
