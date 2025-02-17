""" NOT USE
@font-face {
  font-family: 'Noto Sans JP';
  src: url('fonts/NotoSansJP-Regular.otf') format('opentype');
  font-weight: normal;
  font-style: normal;
}

@font-face {
  font-family: 'Noto Sans SC';
  src: url('fonts/NotoSansSC-Regular.otf') format('opentype');
  font-weight: normal;
  font-style: normal;
}

@font-face {
  font-family: 'Noto Sans TC';
  src: url('fonts/NotoSansSC-Regular.otf') format('opentype');
  font-weight: normal;
  font-style: normal;
}

@font-face {
  font-family: 'Noto Sans HK';
  src: url('fonts/NotoSansHK-Regular.otf') format('opentype');
  font-weight: normal;
  font-style: normal;
}

@font-face {
  font-family: 'Roboto';
  src: url('fonts/Roboto-Regular.ttf') format('opentype');
  font-weight: normal;
  font-style: normal;
}

:root {
  --global-font-family: "Noto Sans JP", "Noto Sans SC", "Noto Sans TC", "Noto Sans HK", "Roboto", Arial, sans-serif;
}

html, body, div, table, tr, td, p, strong, button {
  font-family: var(--global-font-family) !important;
}
"""

custom_css = """
/* Hide sort buttons at gr.DataFrame */
.sort-button {
    display: none !important;
} 

body gradio-app .tabitem .block{
    background: #fff !important;
}

.gradio-container{
    background: #c4c4c440;
}

.tabitem .form{
border-radius: 3px;
}

.main_Header>span>h1{
    color: #fff;
    text-align: center;
    margin: 0 auto;
    display: block;
}

.tab-nav{
    # border-bottom: none !important;
}

.tab-nav button[role="tab"]{
    color: rgb(96, 96, 96);
    font-weight: 500;
    background: rgb(255, 255, 255);
    padding: 10px 20px;
    border-radius: 4px 4px 0px 0px;
    border: none;
    border-right: 4px solid gray;
    border-radius: 0px;
    min-width: 150px;
}

.tabs .tabitem .tabs .tab-nav button[role="tab"]{
    min-width: 90px;
    padding: 5px;
    border-right: 1px solid #186fb4;
    border-top: 1px solid #186fb4;
    border-bottom: 0.2px solid #fff;
    margin-bottom: -2px;
    z-index: 3;
}


.tabs .tabitem .tabs .tab-nav button[role="tab"]:first-child{
    border-left: 1px solid #186fb4;
        border-top-left-radius: 3px;
}

.tabs .tabitem .tabs .tab-nav button[role="tab"]:last-child{
    border-right: 1px solid #186fb4;
}

.tab-nav button[role="tab"]:first-child{
       border-top-left-radius: 3px;
}

.tab-nav button[role="tab"]:last-child{
        border-top-right-radius: 3px;
    border-right: none;
}
.tabitem{
    background: #fff;
    border-radius: 0px 3px 3px 3px !important;
    box-shadow: rgba(50, 50, 93, 0.25) 0px 2px 5px -1px, rgba(0, 0, 0, 0.3) 0px 1px 3px -1px;
}

.tabitem .tabitem{
    border: 1px solid #196fb4;
    background: #fff;
    border-radius: 0px 3px 3px 3px !important;
}

.tabitem textarea, div.tabitem div.container>.wrap{
    background: #f4f8ffc4;
}

.tabitem .container .wrap {
    border-radius: 3px;
}

.tab-nav button[role="tab"].selected{
    color: #fff;
    background: #196fb4;
    border-bottom: none;
}

.tabitem .inner_tab button[role="tab"]{
   border: 1px solid rgb(25, 111, 180);
   border-bottom: none;
}

.app.gradio-container {
  max-width: 1440px;
}

gradio-app{
    background-image: url("https://objectstorage.ap-tokyo-1.oraclecloud.com/n/sehubjapacprod/b/km_newsletter/o/tmp%2Fmain_bg.png") !important;
    background-size: 100vw 100vh !important;
}

input, textarea{
    border-radius: 3px;
}


.container>input:focus, .container>textarea:focus, .block .wrap .wrap-inner:focus{
    border-radius: 3px;
    box-shadow: rgb(255 246 228 / 63%) 0px 0px 0px 3px, rgb(255 248 236 / 12%) 0px 2px 4px 0px inset !important;
    border-color: rgb(249 169 125 / 87%) !important;
}

.tabitem div>button.primary{
    border: none;
    background: linear-gradient(to bottom right, #ffc679, #f38141);
    color: #fff;
    box-shadow: 2px 2px 2px #0000001f;
    border-radius: 3px;
}

.tabitem div>button.primary:hover{
    border: none;
    background: #f38141;
    color: #fff;
    border-radius: 3px;
    box-shadow: 2px 2px 2px #0000001f;
}


.tabitem div>button.secondary{
    border: none;
    background: linear-gradient(to right bottom, rgb(215 215 217), rgb(194 197 201));
    color: rgb(107 106 106);
    box-shadow: rgba(0, 0, 0, 0.12) 2px 2px 2px;
    border-radius: 3px;
}

.tabitem div>button.secondary:hover{
    border: none;
    background: rgb(175 175 175);
    color: rgb(255 255 255);
    border-radius: 3px;
    box-shadow: rgba(0, 0, 0, 0.12) 2px 2px 2px;
}

.cus_ele1_select .container .wrap:focus-within{
    border-radius: 3px;
    box-shadow: rgb(255 246 228 / 63%) 0px 0px 0px 3px, rgb(255 248 236 / 12%) 0px 2px 4px 0px inset !important;
    border-color: rgb(249 169 125 / 87%) !important;
}

input[type="checkbox"]:checked, input[type="checkbox"]:checked:hover, input[type="checkbox"]:checked:focus {
    border-color: #186fb4;
    background-color: #186fb4;
}

#event_tbl{
    border-radius:3px;
}

#event_tbl .table-wrap{
    border-radius:3px;
}

#event_tbl table thead>tr>th{
    background: #bfd1e0;
        min-width: 90px;
}

#event_tbl table thead>tr>th:first-child{
    border-radius:3px 0px 0px 0px;
}
#event_tbl table thead>tr>th:last-child{
    border-radius:0px 3px 0px 0px;
}


#event_tbl table .cell-wrap span{
    font-size: 0.8rem;
}

#event_tbl table{
    overflow-y: auto;
    overflow-x: auto;
}

#event_exp_tbl .table-wrap{
     border-radius:3px;   
}
#event_exp_tbl table thead>tr>th{
    background: #bfd1e0;
}

.count_t1_text .prose{
    padding: 5px 0px 0px 6px;
}

.count_t1_text .prose>span{
    padding: 0px;
}

.cus_ele1_select .container .wrap:focus-within{
    border-radius: 3px;
    box-shadow: rgb(255 246 228 / 63%) 0px 0px 0px 3px, rgb(255 248 236 / 12%) 0px 2px 4px 0px inset !important;
    border-color: rgb(249 169 125 / 87%) !important;
}

.count_t1_text .prose>span{
    font-size: 0.9rem;
}


footer{
  display: none !important;
}

.sub_Header>span>h3,.sub_Header>span>h2,.sub_Header>span>h4{
    color: #fff;
    font-size: 0.8rem;
    font-weight: normal;
    text-align: center;
    margin: 0 auto;
    padding: 5px;
}

@media (min-width: 1280px) {
    .app.svelte-wpkpf6.svelte-wpkpf6:not(.fill_width) {
        max-width: 1400px;
    }
}
.gap.svelte-vt1mxs{
    gap: unset;
}

.tabitem .gap.svelte-vt1mxs{
        gap: var(--layout-gap);
}

@media (min-width: 1280px) {
    .app.svelte-wpkpf6.svelte-wpkpf6:not(.fill_width) {
        max-width: 1400px;
    }
}

"""
