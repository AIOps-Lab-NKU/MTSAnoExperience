<template>
  <div class="home" style="display:flex">
    <left-bar class="left-bar"/>
    <show-right class="middle"/>
    <select-middle class="right"/>
  </div>
</template>

<script>
import leftBar from "../components/leftBar.vue";
import selectMiddle from "../components/selectMiddle.vue";
import showRight from "../components/showRight.vue";

export default {
  name: "homePage",
  components: {
    selectMiddle,
    leftBar,
    showRight
  },
  data() {
    return {
      showDraw: false
    }
  }
}
</script>

<style scoped>
.left-bar {
  display: flex;
  width: 15%;
  height: 100%;
}

.middle {
  width: 55%;
  height: 100%;
  float: left;
}

.right {
  float: left;
  width: 30%;
  height: 100%;
  right: 0;
}

span {
  text-align: center;
  font-size: medium;
  transition: all .3s;
}

span:hover {
  cursor: pointer;
  font-size: larger;
}
</style>
