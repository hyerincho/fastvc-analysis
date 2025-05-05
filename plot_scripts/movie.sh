## These are the parameters to control
tag="061724_fastvc/combineout_ismr" #"080724_fastvc_consistentB"
quantity="blob_analyses" #"log_abs_divB" #"log_beta" #_prims" #"fails" #"floors" #"B2" #"symlog_u^phi_over_uK" #_prims" #"log_rho_poloidal" #"symlog_u^r_over_uff" #"log_Theta" #"log_b" #"symlog_u^th" #_rel" #"log_Gamma" # #"log_K" #"conservation" #"symlog_FE_PAKE_A" #"symlog_FM_A" #"symlog_B3" #"symlog_Beb_over_uff2" #"symlog_u^r_over_uchar" #"log_u" #"log_sigma" #"symlog_u^r" #"log_bsq" #
ghostzone=true #false #
native=true #  false #
embed_label=false #true
overlay_field=false # true #
overlay_streamline=false # true #
overlay_grid=false #true
log_r=false # true #
FIGURES=false #true # 

DATA_DIR="../data"

## Add arguments
args=()
filetype=phdf #rhdf #

if [ $ghostzone == true ]; then
  args+=( '-g' )
fi
if [ $native == true ]; then
  args+=( '--native' )
fi
if [ $embed_label == true ]; then
  args+=( '--embed_label' )
fi
if [ $log_r == true ]; then
  args+=( '--log_r' )
fi
if [ "$overlay_field" == "true" ] ; then
  args+=( '--overlay_field' )
fi
if [ "$overlay_streamline" == "true" ] ; then
  args+=( '--overlay_streamline' )
fi
if [ "$overlay_grid" == "true" ] ; then
  args+=( '--overlay_grid' )
fi
if [[ "$quantity" == *"beta"* ]]; then
    args+=( '--vmin=-1 --vmax=3 --cmap=plasma' )
fi

dir=""
odir="${DATA_DIR}/${tag}/frames_${quantity}"
  
if [ $native == true ]; then
odir="${odir}_native"
fi

#args+=( '--vmin=1e-9 --vmax=1 ' )

pyharm-movie $quantity ${DATA_DIR}/${tag}/${dir}/*.${filetype} --output_dir=$odir ${args[@]} --at=0 #--numeric_fnames
#pyharm-movie $quantity ${DATA_DIR}/${tag}/${dir}/*0.02*.${filetype} --output_dir=$odir ${args[@]} --at=0 --numeric_fnames
