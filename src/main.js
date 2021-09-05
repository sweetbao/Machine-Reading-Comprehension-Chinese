import Vue from 'vue'
import App from './App.vue'
import router from "./router";
import axios from 'axios'

Vue.config.productionTip = false
Vue.prototype.$axios = axios    //把axios挂载到vue的原型中，在vue中每个组件都可以使用axios发送请求
axios.defaults.baseURL = '/api'  //关键代码

new Vue({
  router,
  render: h => h(App),
  components:{
    App
  },
}).$mount('#app')




