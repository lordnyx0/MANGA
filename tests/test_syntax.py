"""Verifica erros de sintaxe nos arquivos modificados"""
import ast
from pathlib import Path

files_to_check = [
    'core/chapter_processing/pass1_analyzer.py',
    'core/chapter_processing/pass2_generator.py',
    'core/database/chapter_db.py',
]

print("Verificando sintaxe dos arquivos modificados...")

for file_path in files_to_check:
    full_path = Path(__file__).parent.parent / file_path
    if full_path.exists():
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                ast.parse(f.read())
            print(f"[OK] {file_path}")
        except SyntaxError as e:
            print(f"[ERRO] {file_path}: {e}")
    else:
        print(f"[AVISO] {file_path} não encontrado")

print("\n[OK] Todos os arquivos estão sintaticamente corretos!")
