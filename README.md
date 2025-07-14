# **oy-gen-pptx**: PowerPoint Presentation Generator

**Author:** oy_plat  
**Version:** 0.0.1  
**Type:** tool  
**Repo:** https://github.com/294033186/oy-gen-pptx  

## Description:
`oy-gen-pptx` is a Dify plugin that allows you to generate PowerPoint presentations from descriptions and templates. The plugin accepts a string input from a large language model (LLM) and optionally allows you to specify an output filename and a custom PowerPoint demo template.

## Features:
- **LLM Input**: Accepts raw content from the LLM to generate the PowerPoint content.
- **Custom Template**: Optionally, input a custom PowerPoint template for styling.
- **Custom Output Filename**: Optionally specify the output file name for the generated PowerPoint.

## Parameters:
1. **llm_str**:
   - Type: string
   - Required: true
   - Description: The content returned by the LLM to generate the PowerPoint.

2. **output_filename**:
   - Type: string
   - Required: false
   - Description: The output filename for the generated PowerPoint presentation.

3. **pptx_demo**:
   - Type: files
   - Required: false
   - Description: Custom PowerPoint template for the generated presentation.

## Usage Example:
1. **config flow**:
   <img width="1439" height="776" alt="image" src="https://github.com/user-attachments/assets/eecd1c78-52d3-4506-bf8c-acd2f9006add" />
2. **Enter prompt words and PPT master**：
   <img width="1434" height="778" alt="image" src="https://github.com/user-attachments/assets/686b44d0-b083-416b-835c-b57efb407948" />
   <img width="1439" height="777" alt="image" src="https://github.com/user-attachments/assets/4a3c5f92-5062-49c7-83c4-950a041a89fa" />

 


