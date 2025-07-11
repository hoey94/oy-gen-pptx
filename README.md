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




