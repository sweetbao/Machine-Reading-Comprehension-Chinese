import Vue from 'vue';
import VueRouter from "vue-router";
const Home = ()=>import('../views/homedir/home')
const Channel = ()=>import('../views/channeldir/channel')
const Active = ()=>import('../views/activedir/active')
const Shop = ()=>import('../views/shopdir/shop')
const Identi = ()=>import('../views/identidir/identi')
const Play = ()=>import('../views/playdir/playVideo')

const routes = [{
    path:'',
    redirect:'/homedir'
},{
    path:'/homedir',
    component:Home
},{
    path:'/channeldir',
    component:Channel
},{
    path:'/activedir',
    component:Active
},{
    path:'/shopdir',
    component:Shop
},{
    path:'/identidir',
    component:Identi
},{
    path:'/playdir',
    component:Play
}
]

Vue.use(VueRouter)
const router = new VueRouter({
    routes:routes,
})
export default router