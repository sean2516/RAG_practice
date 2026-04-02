import fitz  # PyMuPDF

import io
import sys

# 修改标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def detect_and_remove_footer(pdf_path, output_path=None, remove_page_numbers=True):

    doc = fitz.open(pdf_path)
    has_footer = False
    has_page_number = False
    pages_modified = 0
    
    for page in doc:
        text_blocks = page.get_text("blocks")
        footer_blocks = []
        
        for block in text_blocks:
            text = block[4].strip()  
            y0 = block[1]  # 文本顶部位置
            
            # 检测页尾(底部10%区域)
            if y0 > page.rect.height * 0.91 and text:
                footer_blocks.append(block)
                has_footer = True
                
                # 检测页码(包含数字或页码关键词)
                if (any(c.isdigit() for c in text) or 
                    any(mark in text.lower() for mark in ["page", "p.", "页"])):
                    has_page_number = True
        
        
        if remove_page_numbers and footer_blocks:
            for block in footer_blocks:
                rect = fitz.Rect(block[:4])
                page.add_redact_annot(rect, fill=(1, 1, 1))  # 白色填充
                    
            page.apply_redactions()
            pages_modified += 1
    
    # 保存输出
    save_path = output_path if output_path else pdf_path
    doc.save(save_path)
    doc.close()
    
    return has_footer, has_page_number, pages_modified


if __name__ == "__main__":
    input_pdf = "input.pdf"
    output_pdf = "output.pdf"
    
    
    has_footer, has_page_num, modified = detect_and_remove_footer(input_pdf, output_pdf, remove_page_numbers=True)
    
    print(f"检测结果:")
    print(f"- 文档有页尾: {has_footer}")
    print(f"- 页尾包含页码: {has_page_num}")
    print(f"- 已修改页数: {modified}")
    
    if modified > 0:
        print(f"\n已删除页码并保存到: {output_pdf}")
    else:
        print("\n未检测到需要删除的页码")