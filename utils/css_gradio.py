custom_css = """
@font-face {
  font-family: 'Noto Sans JP';
  src: url('fonts/NotoSansJP-Regular.otf') format('opentype');
  font-weight: 400;
  font-style: normal;
}

@font-face {
  font-family: 'Noto Sans SC';
  src: url('fonts/NotoSansSC-Regular.otf') format('opentype');
  font-weight: 400;
  font-style: normal;
}

@font-face {
  font-family: 'Noto Sans TC';
  src: url('fonts/NotoSansSC-Regular.otf') format('opentype');
  font-weight: 400;
  font-style: normal;
}

@font-face {
  font-family: 'Noto Sans HK';
  src: url('fonts/NotoSansHK-Regular.otf') format('opentype');
  font-weight: 400;
  font-style: normal;
}

@font-face {
  font-family: 'Roboto';
  src: url('fonts/Roboto-Regular.ttf') format('truetype');
  font-weight: 400;
  font-style: normal;
}

:root {
  --global-font-family: 
    "Noto Sans JP", 
    "Noto Sans SC", 
    "Noto Sans TC", 
    "Noto Sans HK", 
    "Roboto", 
    Arial, 
    sans-serif;
  --primary-color: #196fb4;
  --secondary-color: #f38141;
  --text-light: #fff;
  --shadow-sm: 0 2px 5px -1px rgba(50, 50, 93, 0.25), 0 1px 3px -1px rgba(0, 0, 0, 0.3);
}

/* ======= Base Styles ======= */
html,
body,
div,
table,
tr,
td,
p,
strong,
button {
  font-family: var(--global-font-family) !important;
}

input,
textarea {
  border-radius: 3px;
}

/* ======= Component Styles ======= */
.app {
  background: #c4c4c440;
}

/* Header Styles */
.main_Header > span > h1 {
  color: var(--text-light);
  text-align: center;
  margin: 0 auto;
  display: block;
  overflow: hidden;
}

.sub_Header > span > h3,
.sub_Header > span > h2,
.sub_Header > span > h4 {
  color: var(--text-light);
  font-size: 0.8rem;
  font-weight: 400;
  text-align: center;
  margin: 0 auto;
  padding: 5px;
}

/* Tab Styles */
.tabs {
  background: var(--text-light);
  border-radius: 0 3px 3px 3px !important;
  box-shadow: var(--shadow-sm);
  gap: unset;
}

.tab-container {
  button[role="tab"] {
    color: #606060;
    font-weight: 500;
    background: var(--text-light);
    padding: 10px 20px;
    border: none;
    border-right: 4px solid gray;
    min-width: 150px;
    
    &.selected {
      color: var(--text-light);
      background: var(--primary-color);
      border-bottom: none;
    }
    
    &:last-child {
      border-right: none;
      border-top-right-radius: 3px;
    }
    
    &:first-child {
      border-top-left-radius: 3px;
    }
  }
}

/* Table Styles */
#event_tbl {
  .table-wrap {
    border-radius: 3px;
  }
  
  thead > tr > th {
    background: #bfd1e0;
    min-width: 90px;
    
    &:first-child { border-radius: 3px 0 0 0; }
    &:last-child { border-radius: 0 3px 0 0; }
  }
  
  .cell-wrap span {
    font-size: 0.8rem;
  }
}

/* Button Styles */

/* Form Elements */
.container {
  input:focus,
  textarea:focus,
  .wrap .wrap-inner:focus {
    border-color: rgb(249 169 125 / 87%) !important;
    border-radius: 3px;
    box-shadow: 
      rgb(255 246 228 / 63%) 0 0 0 3px,
      rgb(255 248 236 / 12%) 0 2px 4px 0 inset !important;
  }
}

/* Responsive Styles */
@media (min-width: 1280px) {
  .app.gradio-container:not(.fill_width) {
    max-width: 1400px;
  }
}

/* Accessibility */
footer { display: none !important; }
.sort-button { display: none !important; }

gradio-app {
    background-image: url("https://objectstorage.ap-tokyo-1.oraclecloud.com/n/sehubjapacprod/b/km_newsletter/o/tmp%2Fmain_bg.png") !important;
    background-size: 100vw 100vh !important;
}
"""
