variable "availability_domain" {
  default = "bxtG:AP-TOKYO-1-AD-1"
}

variable "compartment_ocid" {
}

variable "instance_display_name" {
  default = "aipoc-instance"
}

variable "instance_shape" {
  default = "VM.Standard.E4.Flex"
}

variable "instance_flex_shape_ocpus" {
  default = 2
}

variable "instance_flex_shape_memory" {
  default = 16
}

variable "instance_boot_volume_size" {
  default = 100
}

variable "instance_boot_volume_vpus" {
  default = 20
}

variable "instance_image_source_id" {
  default = "ocid1.image.oc1.ap-tokyo-1.aaaaaaaa2vfxs5eqzyk4o7lyd6j4bo42pfy6oqrwfzkkho54uwiafy7nxu3a"
}

variable "instance_user_data" {
  default = ""
}

variable "subnet_ai_subnet_id" {
  default = "ocid1.subnet.oc1.ap-tokyo-1.aaaaaaaa5imetr2dm3u5pxdyzk63hku5struudgqxrm5glvgpjfgspa64gba"
}

variable "security_list_id" {
  default = "ocid1.securitylist.oc1.ap-tokyo-1.aaaaaaaa3ffult7vy2nfc2bf5pqkio3vowc2psriqdj3w5byc7xbyumfilqq"
}

variable "ssh_authorized_keys" {
}