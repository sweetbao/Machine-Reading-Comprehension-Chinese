<template>
    <swiper ref="mySwiper" :options="swiperOptions">
        <swiper-slide>
            <img class="swip-img" src="../../assets/img/swiper_1.png">
        </swiper-slide>
        <swiper-slide>
            <img class="swip-img" src="../../assets/img/swiper_2.png">
        </swiper-slide>
        <swiper-slide>
            <img class="swip-img" src="../../assets/img/swiper_3.png">
        </swiper-slide>
        <swiper-slide>
            <img class="swip-img" src="../../assets/img/swiper_4.png">
        </swiper-slide>
        <swiper-slide>
            <img class="swip-img" src="../../assets/img/swiper_5.png">
        </swiper-slide>
        <div class="swiper-pagination"  slot="pagination"></div>
        <div class="swiper-scrollbar"   slot="scrollbar"></div>
    </swiper>
</template>

<script>
    import {Swiper,SwiperSlide,directive} from 'vue-awesome-swiper'
    import 'swiper/swiper-bundle.css'
    export default {
        name: "VueSwiper",
        components:{
            Swiper,
            SwiperSlide
        },
        directives: {
            swiper: directive
        },
        data(){
            return{
                swiperOptions:{
                    navigation: {
                        nextEl: '.swiper-button-next',
                        prevEl: '.swiper-button-prev',
                    },
                    autoplay: {
                        delay: 1000,
                        stopOnLastSlide: true,
                    },
                    loop:true,
                    pagination:{
                        el:'.swiper-pagination',
                        clickable:true,
                    }
                }
            }
        },
        computed: {
            swiper() {
                return this.$refs.mySwiper.$swiper
            }
        },
        mounted() {
            console.log('this is current swiper instance object', this.swiper)
            this.swiper.slideTo(1, 3000, false)
        }
    }
</script>

<style scoped>

    .swip-img{
        height: 170px;
        width: 96%;
        margin: 10px;
        border-radius: 5px;
    }
</style>