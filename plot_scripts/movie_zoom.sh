## These are the parameters to control
tag="052825_a0.9_rB2e5_bondi_eks_largerout" #"043025_a0.9_rB2e5_bondi_eks"
fnum="04783" #"03830"
quantity="log_Theta" #"log_rho" #

DATA_DIR="../data"

## Add arguments
args=()
filetype=phdf #rhdf #

if [[ "$quantity" == *"beta"* ]]; then
    args+=( '--vmin=-1 --vmax=3 --cmap=plasma' )
fi

if [[ "$quantity" == *"log_rho"* ]]; then
    args+=( '--vmin=-10 --vmax=-4 --cmap=turbo')
fi
if [[ "$quantity" == *"log_Theta"* ]]; then
    args+=( '--vmin=-5.096910013008056 --vmax=-1 --cmap=gist_heat')
fi

dir=""
odir="${DATA_DIR}/${tag}/frames_zoom_${quantity}"

base=1.1 #1.2
for i in {30..160}
do
    sz=$(echo "scale=10; e($i * l($base))" | bc -l)
    ipad=$(printf "%05d" $i)
    echo $sz
    pyharm-movie $quantity ${DATA_DIR}/${tag}/${dir}/*${fnum}*.${filetype} --output_dir=$odir ${args[@]} --at=0 --sz=$sz --frame_name="frame_${ipad}.png" --scalebar
done
