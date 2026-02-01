CUSTOM_CSS = """
    .gradio-container{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen,Ubuntu,Cantarell,'Open Sans','Helvetica Neue',sans-serif;}
    progress{width:100%;height:20px;margin:4px 0;}
    #train-log-box,
    #sft-log-box{
        height:450px;
        overflow-y:auto !important;
        display:block !important;
        font-family:monospace;
        padding:8px;
        background:white !important;
        border:1px solid #ddd !important;
        white-space:pre-wrap;
        word-break:break-word;
        scroll-behavior:smooth;
    }
    #train-log-box > div,
    #sft-log-box > div{
        background:transparent !important;
    }
    #sft-dataset-example{
        height:240px;
        display:flex;
        flex-direction:column;
    }
    #sft-dataset-example .cm-editor{
        flex:1;
        min-height:0;
    }
    #sft-dataset-example .cm-scroller{
        overflow:auto;
    }
    #inf-result-html{
        background:white !important;
    }
    #inf-result-html > div{
        background:white !important;
    }
    """

TOKEN_COLORS = [
    "#FFE066",  # Yellow
    "#98D8AA",  # Light green
    "#87CEEB",  # Sky blue
    "#DDA0DD",  # Plum
    "#F0B27A",  # Light orange
    "#AED6F1",  # Light blue
    "#F9E79F",  # Light yellow
    "#D5A6BD",  # Light purple
]
