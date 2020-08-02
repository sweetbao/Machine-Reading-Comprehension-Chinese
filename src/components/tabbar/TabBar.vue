<template>
    <div id="tab-bar">
        <div class="tab-bar-item" @click="clickhome">
            <img v-if="isActive1" src="../../assets/img/home_active.png" alt="">
            <img v-else src="../../assets/img/home.png" alt="">
            <div :class="{active: isActive1}" >首页</div>
        </div>
        <div class="tab-bar-item" @click="clickchannel">
            <img v-if="isActive2" src="../../assets/img/channel_active.png" alt="">
            <img v-else src="../../assets/img/channel.png" alt="">
            <div :class="{active: isActive2}">频道</div>
        </div>
        <div class="tab-bar-item" @click="clickactive">
            <img v-if="isActive3" src="../../assets/img/active_active.png" alt="">
            <img v-else src="../../assets/img/active.png" alt="">
            <div :class="{active: isActive3}">动态</div>
        </div>
        <div class="tab-bar-item" @click="clickshop">
            <img v-if="isActive4" src="../../assets/img/shop_active.png" alt="">
            <img v-else src="../../assets/img/shop.png" alt="">
            <div :class="{active: isActive4}">会员购</div>
        </div>
        <div class="tab-bar-item" @click="clickidenti">
            <img v-if="isActive5" src="../../assets/img/identi_active.png" alt="">
            <img v-else src="../../assets/img/identi.png" alt="">
            <div :class="{active: isActive5}">我的</div>
        </div>
    </div>
</template>

<script>
    export default {
        name: "TabBar",
        data(){
            return{
                path:'/homedir',
            }
        },
        computed:{
            isActive1(){
                return this.$route.path.indexOf('/homedir') !== -1
            },
            isActive2(){
                return this.$route.path.indexOf('/channeldir') !== -1
            },
            isActive3(){
                return this.$route.path.indexOf('/activedir') !== -1
            },
            isActive4(){
                return this.$route.path.indexOf('/shopdir') !== -1
            },
            isActive5(){
                return this.$route.path.indexOf('/identidir') !== -1
            },
        },
        methods:{
            clickhome(){
                this.$router.replace('/homedir')
            },
            clickchannel(){
                this.$router.replace('/channeldir')
            },
            clickactive(){
                this.$router.replace('/activedir')
            },
            clickshop(){
                this.$router.replace('/shopdir')
            },
            clickidenti(){
                this.$router.replace('/identidir')
            },

        }
    }
</script>

<style scoped>
    #tab-bar{
        background-color: white;
        display: flex;
        position: fixed;
        left: 0;
        right: 0;
        bottom: 0;
    }
    .tab-bar-item{
        flex: 1;
        text-align: center;
        height: 55px;

    }
    .tab-bar-item img{
        width: 28px;
        height: 28px;
        vertical-align: middle;
    }
    .tab-bar-item div{
        font-size: 8px;
    }
    .active{
        color: palevioletred;
    }
</style>