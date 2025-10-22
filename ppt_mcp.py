from fastmcp import FastMCP
import app

mcp = FastMCP("生成pptx mcp", host='172.22.61.12', port=8001)

@mcp.tool()
def ppt_generate(template_name, pptx_json_str, output_file_name) -> int:
    """
    根据pptx和json内容生成ppt
    template_name: 模版名称
    pptx_json_str: ppt 内容包含标题和要点 
    output_file_name: 输出文件名称
    """
    
    if template_name is None:
        template_name = 'default'
    
    app.generate_pptx(template_name, pptx_json_str,output_file_name)
    return f'http://172.22.61.12:8000/{output_file_name}.pptx'


if __name__ == "__main__":
    mcp.run(transport='http')