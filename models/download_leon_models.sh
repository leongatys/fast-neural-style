BASE_URL="http://bethgelab.org/media/uploads/stylecontrol/feedforwardmodels/"

mkdir -p trained
cd trained
curl -O "$BASE_URL/candy_lum_256-lum.t7"
curl -O "$BASE_URL/candy_lum_256-lum.json"
curl -O "$BASE_URL/candy_over_feathers_256_guidance_sw_5.0_slim.t7"
curl -O "$BASE_URL/candy_over_feathers_256_guidance_sw_5.0_slim.json"
cd ../../

