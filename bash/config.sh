#config.sh

#This is free and unencumbered software released into the public domain.

#source this file to access the following functions
#they parse and write config files in
#key=value format
#for each function, the first argument is an associative array
# and the second argument is the filename

#e.g.
#declare -A cfg
#parse_config cfg config_file.ini
#echo ${cfg[key_name]}
#to check if a key exists:
# [[ -z ${cfg[key_name]} ]]

parse_config() {
	#set internal field separator to split around =
	local IFS="= "
	#loop over file:
	while read -r name value
	do
		#strip leading and trailing whitespace:
		name=$(echo $name | sed -e "s/^[[:space:]]*//" -e "s/[[:space:]]*$//")
		value=$(echo $value | sed -e "s/^[[:space:]]*//" -e "s/[[:space:]]*$//")
		if [[ -n $name ]]; then
			#use declare to set the value in the global scope
			declare -g "$1[\"$name\"]=$value"
		fi
	done < $2
}
write_config() {
	local arr_keys="\${!$1[@]}"
	local arr_val="\${$1[\"\$k\"]}"
	for k in $(eval echo $arr_keys); do
		printf "%s=%s\n" "$k" "$(eval echo $arr_val)"
	done > $2
}