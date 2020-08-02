<template>
    <div class="tab-control">
        <div v-for="(item,index) in titles"
             class="tab-control-item" :class="{active: index === currentIndex}" @click="itemclick(index)">
            <span>{{item}}</span>
        </div>
    </div>
</template>

<script>
    export default {
        name: "TabControl",
        props:{
            titles:{
                type:Array,
                default(){
                    return []
                }
            }
        },
        data(){
            return{
                currentIndex:0
            }
        },
        methods:{
            itemclick(index){
                this.currentIndex = index;
            }
        }
    }
</script>

<style scoped>
    .tab-control{
        display: flex;
        text-align: center;
        background-color: #ffffff;
        border-bottom: 1px solid #dcdcdc;
    }
    .tab-control-item{
        flex: 1;
        height: 40px;
        line-height: 40px;

    }
    .tab-control-item span{
        padding: 4px;
        font-size: 15px;
    }
    .active{
        color: palevioletred;

    }
    .active span{
        border-bottom: 3px solid palevioletred;

    }
</style>