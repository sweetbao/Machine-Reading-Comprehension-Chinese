<template>
    <div class="top-nav">
        <div class="user-icon">
            <img src='../../assets/img/userIcon.jpeg'>
        </div>
        <div class="search">
<!--            <i class="fa fa-search" id="i-advanced-search-i"></i>-->
            <form action="javascript:return true;" @submit.prevent>
                <input ref="searchInput"
                       type="search"
                       id="i-advanced-search"
                       placeholder="  输入视频链接"
                       @keyup.13="searchurl($event)"
                >
            </form>

        </div>
        <div class="game-icon">
            <img src='../../assets/img/gamebutton.png'>
        </div>
        <div class="message-icon">
            <img src='../../assets/img/messagebutton.png'>
        </div>
    </div>

</template>

<script>
    export default {
        name: "TopNav",
        data(){
            return{
                search_url:''
            }

        },
        methods:{
            searchurl(e){
                var keyCode = window.Event ? e.keyCode : e.which;
                //console.log("回车搜索", keyCode, e);
                this.search_url = e.target.value;
                console.log(this.search_url);
                this.$router.push({
                    path: '/playdir',
                    query: {
                        vurl:this.search_url,
                    }
                })
            }
        }
    }

</script>

<style scoped>
    .top-nav{
        display: flex;
        flex: 1;
    }
    .user-icon img{
        width: 50px;
        height: 50px;
        margin-left: 5px;
    }
    .game-icon img{
        width: 33px;
        height: 33px;
        margin-top: 17px;
    }
    .message-icon img{
        width: 33px;
        height: 33px;
        margin-top: 17px;
        flex: 1;
        text-align: right;
    }
    .search{
        width: 200px;
        margin:20px 8%;
    }
    #i-advanced-search{
        width: 180px;
        height: 30px;
        border-radius: 20px;
        background-color: #eeeeee;
        border-style: none;
    }

</style>