<template>
    <div class="video-recom">
        <video-item v-for="item in videos" :video-item="item"></video-item>

    </div>
</template>

<script>
    import VideoItem from "./VideoItem";
    export default {
        name: "VideoRecom",
        components: {VideoItem},
        props:{
            videos:{
                type:Array,
                default(){
                    return []
                }
            }
        }
    }
</script>

<style scoped>
    .video-recom{
        display: flex;
        flex-wrap: wrap;
        justify-content: space-around;
    }


</style>