"""
Gera ícones PNG a partir dos SVGs.
Execute: python generate_icons.py
"""

try:
    from PIL import Image, ImageDraw, ImageFont
    import io
    
    sizes = [16, 48, 128]
    
    for size in sizes:
        # Cria imagem com gradiente
        img = Image.new('RGBA', (size, size), (102, 126, 234, 255))
        draw = ImageDraw.Draw(img)
        
        # Gradiente simples
        for y in range(size):
            r = int(102 + (118 - 102) * y / size)
            g = int(126 + (75 - 126) * y / size)
            b = int(234 + (162 - 234) * y / size)
            draw.line([(0, y), (size, y)], fill=(r, g, b, 255))
        
        # Bordas arredondadas
        from PIL import ImageFilter
        
        # Salva
        img.save(f'icons/icon{size}.png')
        print(f'Generated icon{size}.png')
        
    print('Done!')
    
except ImportError:
    print('PIL not available. Please install: pip install Pillow')
    print('Or manually convert SVG files to PNG.')
