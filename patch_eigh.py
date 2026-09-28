import os
import glob

directory = r'c:\temp\ECG_Benchmark\src'
files = glob.glob(directory + '/**/*.py', recursive=True)
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    
    # We must also clean up the manual SafeEigh we added in adaptive_ptco.py
    if 'class SafeEigh(torch.autograd.Function):' in content and 'adaptive_ptco.py' in file.replace('\\', '/'):
        # Just manually handle this if needed, or we just rely on safe_eigh call
        # Actually, adaptive_ptco already uses SafeEigh.apply, let's also replace that
        content = content.replace('SafeEigh.apply(', 'safe_eigh(')
        modified = True
        
    if 'torch.linalg.eigh(' in content:
        content = content.replace('torch.linalg.eigh(', 'safe_eigh(')
        modified = True
        
    if modified:
        if 'from src.utils.safe_eigh import safe_eigh' not in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('import') or line.startswith('from'):
                    lines.insert(i, 'from src.utils.safe_eigh import safe_eigh')
                    break
            content = '\n'.join(lines)
        with open(file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Updated {file}')
