<template>
    <div class="video-item" @click="videoClick">
        <img :src="videoItem.vpic" alt="">
        <div class="video-info">
            <p>{{videoItem.vt}}</p>
            <span class="videoitem-classify">{{videoItem.shortTitle}}</span>
        </div>
    </div>
</template>

<script>
    export default {
        name: "VideoItem",
        props:{
            videoItem:{
                type:Object,
                default(){
                    return{}
                }
            }
        },
        methods:{
            videoClick() {
                //this.$router.push('/playdir')
                this.$router.push({
                    path: '/playdir',
                    query: {
                        vurl:this.videoItem.vurl
                    }
                })
            }
        }

    }
</script>

<style scoped>
    .video-item {
        padding-bottom: 40px;
        position: relative;
        width: 45%;
    }

    .video-item img {
        width: 100%;
        border-radius: 5px;
    }

    .video-info {
        font-size: 12px;
        position: absolute;
        bottom: 5px;
        left: 0;
        right: 0;
        overflow: hidden;
        text-align: center;
    }

    .video-info p {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        margin-bottom: 3px;
    }

    .videoitem-classify {
        color: darkgray;
        margin-right: 20px;
    }

    /*.goods-info .collect {*/
    /*    position: relative;*/
    /*}*/

    /*.goods-info .collect::before {*/
    /*    content: '';*/
    /*    position: absolute;*/
    /*    left: -15px;*/
    /*    top: -1px;*/
    /*    width: 14px;*/
    /*    height: 14px;*/
    /*    background: url("~assets/img/common/collect.svg") 0 0/14px 14px;*/
    /*}*/
</style>