import time
import streamlit as st
from hydralit import HydraHeadApp
from hydralit_components import HyLoader, Loaders


class MyLoadingApp(HydraHeadApp):

    def __init__(self, title = 'Loader', delay=0,loader=Loaders.standard_loaders, **kwargs):
        self.__dict__.update(kwargs)
        self.title = title
        self.delay = delay
        self._loader = loader

    def run(self,app_target):

##        se_loader_txt = """
##            <style> 
##            #rcorners1 {
##              border-radius: 25px;
##              background: grey;
##              color: #00000;
##              alignment: center;
##              opacity: 0.95;
##              padding: 20px; 
##              width: 1920px;
##              height: 400px; 
##              z-index: 9998; 
##            }
##            #banner {
##              color: white;
##              vertical-align: text-top;
##              text-align: center;
##              z-index: 9999; 
##            }
##            </style>
##            <div id="rcorners1">
##            <h1 id="banner">Now loading</h1>
##            <br>
##            </div>
##            """
        
        with HyLoader("Now loading", loader_name=self._loader,index=[3]):
            time.sleep(int(self.delay))
            app_target.run()

        

