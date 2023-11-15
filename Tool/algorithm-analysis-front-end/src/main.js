import {createApp} from 'vue'
import App from './App.vue'
import * as echarts from 'echarts';
import Store from "./store/index"
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import axios from 'axios'
import Dygraphs from 'dygraphs'
import {showFullScreenLoading, tryHideFullScreenLoading} from "./js/animate";
import {router} from './router'


const app = createApp(App);
app.config.globalProperties.$finishLoading = tryHideFullScreenLoading
app.config.globalProperties.$showLoading = showFullScreenLoading
app.config.globalProperties.$echarts = echarts
app.config.globalProperties.$dygraphs = Dygraphs
app.config.globalProperties.$http = axios
app.use(Store)
app.use(router)
app.use(ElementPlus)
app.mount('#app')
