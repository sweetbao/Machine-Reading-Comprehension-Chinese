<template>
    <div>
        <analyze-video :video-src="videoUrl" ></analyze-video>
    </div>

</template>

<script>

    import AnalyzeVideo from "./analyzeVideo";
    export default {
        name: "playVideo",
        components: {AnalyzeVideo},
        data(){
            return{
                videoUrl:null
            }
        },
        created() {
            this.videoUrl = this.$route.query.vurl;
            console.log(this.$route.query.vurl);
        }
    }
</script>

<style scoped>

</style>