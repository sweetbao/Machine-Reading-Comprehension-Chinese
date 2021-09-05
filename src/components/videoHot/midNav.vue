<template>
    <div class="mid-nav">
        <div class="mid-nav-item">
            <img src="../../assets/img/hotorder.png">
            <img src="../../assets/img/hotweek.png">
            <img src="../../assets/img/hotfood.png">
            <img src="../../assets/img/hotgame.png">
            <img src="../../assets/img/hotnece.png">

        </div>
    </div>
</template>

<script>
    export default {
        name: "midNav",
        data(){
          return{
              hotClassify:['排行榜','每周必看','美食','游戏','入站必刷'],
          }
        },
        props: {
            hot_list:{
                type: Array,
                default(){
                    return []
                }
            }
        }
    }
</script>

<style scoped>
    .mid-nav{
        display: flex;
    }
    .mid-nav-item img{
        width: 12%;
        text-align: center;
        margin:2px 4%
    }
</style>