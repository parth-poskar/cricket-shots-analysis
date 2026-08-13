import os
for root, dirs, files in os.walk('.'):
    if 'venv' in root or '.git' in root: continue
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                with open(path, 'r', encoding='utf-8') as f_in:
                    content = f_in.read()
                if '->' in content:
                    with open(path, 'w', encoding='utf-8') as f_out:
                        f_out.write(content.replace('->', '->'))
                    print(f"Fixed {path}")
            except Exception as e:
                pass
