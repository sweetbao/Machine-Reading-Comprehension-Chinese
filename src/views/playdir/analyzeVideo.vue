<template>
    <div id="analyze-video">
        <div class="api-select">
            <div class="apiList">
                <span @click.stop="toggleList">接口{{ getNowApiName }}</span>
                <ul v-if="isShow">
                    <li v-for="(item,index) in api" @click="selectApi(index)">接口 {{ index+1 }}</li>
                </ul>
            </div>
        </div>
        <div class="video-div">
            <iframe id="videos" src="" frameborder="0" :width="width" :height="height"></iframe>
        </div>

    </div>

</template>

<script>
    import $ from 'jquery'
    export default {
        name: "analyzeVideo",
        props:{
            videoSrc: {
                type:String,
                default(){
                    return null
                }
            }
        },
        data(){
            return{
                api:[
                    "http://2gty.com/apiurl/yun.php?url=",
                    "http://vip.jlsprh.com/index.php?url=",
                    "http://mlxztz.com/player.php?url=",
                    "http://jx.ejiafarm.com/dy.php?url=",
                    "http://api.baiyug.cn/vip/index.php?url=",
                    "http://api.pucms.com/?url=",
                    "http://000o.cc/jx/ty.php?url=",
                    "http://jiexi.92fz.cn/player/vip.php?url=",
                    "http://www.662820.com/xnflv/index.php?url=",
                    "http://aikan-tv.com/?url=",

                    "http://www.82190555.com/index/qqvod.php?url=",
                    "http://api.pucms.com/?url=",
                    "http://j.zz22x.com/jx/?url=",
                    "https://api.flvsp.com/?url=",
                    "http://api.xfsub.com/index.php?url=",
                    "http://65yw.2m.vc/chaojikan.php?url=",
                    "http://www.82190555.com/index/qqvod.php?url=",
                    "http://jx.ejiafarm.com/yun.php?url=",
                    "http://vip.jlsprh.com/index.php?url=",

                    "http://jqaaa.com/jx.php?url=",
                    "http://jiexi.92fz.cn/player/vip.php?url=",
                    "http://api.nepian.com/ckparse/?url=",
                    "http://j.zz22x.com/jx/?url=",
                    "http://www.efunfilm.com/yunparse/index.php?url=",
                    "http://jx.ejiafarm.com/jx1/dy1.php?url=",
                    "http://api.47ks.com/webcloud/?v=",
                ],
                nowIndex:0,
                //videoSrc:"https://v.qq.com/x/cover/p8bvvfh82dqrqgq.html",
                isShow:false,
                width:"",
                height:"",
            }
        },
        methods:{
            toggleList:function(){
                if(this.isShow == false){
                    this.isShow = true;
                } else {
                    this.isShow = false;
                }
            },
            selectApi:function(index){
                this.nowIndex = index;
                this.toggleList();
            },
        },
        computed:{
            getNowApiName:function(){
                return this.nowIndex +1 ;
            },
            getApi:function(){
                return this.api[this.nowIndex] + this.videoSrc ;
            }
        },
        watch:{
            videoSrc:function(){
                document.getElementById("videos").src = this.getApi;
            },
            nowIndex:function(){
                document.getElementById("videos").src = this.getApi;
            }
        },
        directives:{
            focus:{
                inserted:function(el){
                    el.focus();
                }
            }
        },
        mounted:function(){
            let that = this;
            // 点击空白区域，列表消失
            document.documentElement.addEventListener("click",function(){
                that.isShow = false;
            })
            // 计算屏幕长宽
            this.width = $(window).width()+ "px";
            this.height = ($(window).height() -60) + "px";
            // 打开默认播放默认视频
            document.getElementById("videos").src = this.getApi;
        }
    }

</script>

<style scoped>
    * {
        padding: 0;
        margin: 0;
    }
    #analyze-video {
        height: 100%;
        width: 100%;
        background-color: #000000;
        padding-top:0px ;
        position: relative;
    }
    span,ul{
        user-select: none;
    }
    .api-select {
        width: 100%;
        height: 30px;
        line-height: 30px;
        /*border:4px solid #fff;*/
        border-radius: 3px;
        position: absolute;
        margin:auto;
        padding-top: 5px;
        padding-right:0px;
        z-index: 2;
    }
    .api-select input {
        width: 50%;
        height: 30px;
        border: none;
        outline: none;
        text-indent: 1em;
        position: absolute;
    }

    .api-select .apiList{
        position: absolute;
        right: 0;
        width: 100px;
        line-height: 30px;
        text-align: center;
        overflow-x: hidden;
    }
    .api-select .apiList span{
        width: 100px;
        display: block;
        cursor: pointer;
        position: relative;
        background-color: gainsboro;
        border-radius: 10px;
    }
    .api-select .apiList span:after{
        content:"";
        display: block;
        position: absolute;
        top: 11px;
        right: 8px;
        width: 0;
        height: 0;
        border-left: 6px solid transparent;
        border-right: 6px solid transparent;
        border-top: 8px solid black;
        border-bottom: none;

    }
    .api-select .apiList ul {
        background-color: #fff;
        margin-top: 4px;
        list-style-type: none;
        width: 120px;
        height: 300px;
        overflow-y: scroll;
    }
    .api-select .apiList ul li {
        cursor: pointer;
    }
    .api-select .apiList ul li:hover {
        background-color: rgba(255,0,0,.5);
    }
    .api-select .tips {
        position: absolute;
        right: -20px;
        color: #fff;
        transform:translateX(100%);
    }
    .video-div{
        position: absolute;
        width: 100%;
    }
</style>