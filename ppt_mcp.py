from fastmcp import FastMCP
import app

mcp = FastMCP("生成pptx mcp", host='172.22.61.12', port=8001)

@mcp.tool()
def add(pptx_json_str,output_file_name) -> int:
    """根据pptx和json内容生成ppt"""
    return app.generate_pptx(pptx_json_str,output_file_name)

if __name__ == "__main__":
    mcp.run(transport='sse')