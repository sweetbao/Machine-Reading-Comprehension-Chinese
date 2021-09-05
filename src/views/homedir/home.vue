<template>
    <div id="home">
        <top-nav class="top-nav"></top-nav>
        <tab-control class="tab-control"
                     :titles="['直播','推荐','热门','追番','影视','说唱区']"
                     @tabClick = "tabClick"
        ></tab-control>
        <div class="recom-class" v-if="isRecom">
            <vue-swiper></vue-swiper>
            <video-recom :videos="vlist"></video-recom>
        </div>
        <div class="hot-class" v-if="isHot">
            <video-hot :videos="hotlist"></video-hot>
        </div>

    </div>

</template>

<script>
    import TabControl from "../../components/tabControl/TabControl";
    import TopNav from "../../components/topNav/TopNav";
    import VueSwiper from "../../components/vueSwiper/VueSwiper";
    import axios from "axios";
    import VideoRecom from "../../components/videoRecom/VideoRecom";
    import VideoHot from "../../components/videoHot/VideoHot";

    //import {getHomeMultidata,getHomeGoods} from "../../components/netWork/request";

    export default {
        name: "home",
        components:{
            VideoHot,
            VideoRecom,
            TopNav,
            TabControl,
            VueSwiper
        },
        data(){
            return{
                videos:{
                    page:0,
                    list:[],
                },
                result: null,
                vlist:[],
                mydata:'',
                hotlist:[],
                isRecom :1,
                isHot :0,
            }
        },
        methods:{
            tabClick(index){
                console.log(index);
                switch (index) {
                    case 1:
                        this.isRecom = 1
                        this.isHot = 0
                        break
                    case 2:
                        this.isHot = 1
                        this.isRecom = 0
                        break
                }
                //this.$router.replace('/homedir/homeHotdir')
            }
        },
        created() {
            //getHomeMultidata().then(res =>{})
            // getHomeGoods(1).then(res =>{
            //     console.log(res);
            // })
            axios({
                //url:'httpbin.org/',
                //url:'http://cache.video.iqiyi.com/jp/avlist/202861101/1/?callback=jsonp9',
                url: '/avlist/202861101/1/?callback=jsonp9',
                method: 'get'
                }).then(res => {
                    this.mydata = res.data.slice(11,-15);
                    //console.log(this.mydata);
                    var outlist = JSON.parse(this.mydata).data.vlist;
                    var outarr = JSON.stringify(outlist);
                    this.vlist = JSON.parse(outarr);
                    console.log(this.vlist)
                })
            axios({
                url: '/web-interface/dynamic/region?&;;jsonp=jsonp&ps=10&rid=1',
                method: 'get'
            }).then(res2 => {
                console.log(res2)
            })

        }
    }
</script>

<style scoped>
    .tab-control{
        position: sticky;
        top: 0px;
        z-index: 999;
    }
</style>