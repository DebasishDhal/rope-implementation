import gradio as gr 
import spaces

@spaces.GPU
def gpu_test():
    return "GPU available"


with gr.Blocks(title="RoPE Explorer") as demo: 
    with gr.Tabs(): 
        with gr.Tab("Block 01"): 
            gr.Markdown("## Block 01") 
            gr.Markdown("Placeholder for the first visualization.") 
        
        with gr.Tab("Block 02"): 
            gr.Markdown("## Block 02") 
            gr.Markdown("Placeholder for the second visualization.") 
            
if __name__ == "__main__": 
    demo.launch( server_name="0.0.0.0", server_port=7860, )