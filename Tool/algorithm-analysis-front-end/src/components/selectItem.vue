<template>
  <div class="selectButton">
    <div class="p-checkbox-box" @click="onClick">
      <template v-if="isSelected">
        <el-icon size="20px">
          <check/>
        </el-icon>
      </template>
    </div>
  </div>
</template>

<script>
import {Check} from "@element-plus/icons-vue";

export default {
  name: "selectItem",
  props: {
    index: Number,
  },
  data() {
    return {
      isSelected: false
    }
  },
  components: {
    Check
  },
  computed: {
    getIsClear() {
      return this.$store.state.isCleared
    }
  },
  watch: {
    getIsClear: {
      deep: true,
      handler(val) {
        if (val) {
          this.isSelected = false;
        }
      }
    }
  },
  methods: {
    onClick() {
      this.isSelected = !this.isSelected
      this.$showLoading()
      setTimeout(() => {
        if (this.isSelected) {
          this.$store.commit("addSelect", this.index)
        } else {
          this.$store.commit("delSelect", this.index)
        }
      }, 100)
    }
  }
}
</script>

<style scoped>
.selectButton {
  width: 25px;
  height: 25px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.p-checkbox-box {
  border: 2px solid #ced4da;
  background: #ffffff;
  width: 20px;
  height: 20px;
  color: #212529;
  border-radius: 4px;
  transition: all .3s;
}

.p-checkbox-box:hover {
  cursor: pointer;
}

.p-checkbox .p-checkbox-box {
  transition-duration: 0.3s;
  color: #ffffff;
  font-size: 14px;
}

.pi {
  font-family: primeicons;
  speak: none;
  font-style: normal;
  font-weight: 400;
  font-variant: normal;
  text-transform: none;
  line-height: 1;
  display: inline-block;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
</style>
