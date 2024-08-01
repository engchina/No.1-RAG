variable "availability_domain" {
  default = "bxtG:AP-TOKYO-1-AD-1"
}

variable "compartment_ocid" {
  default = ""
}

variable "adb_name" {
  default = "ADB"
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
  default = "ocid1.image.oc1.ap-tokyo-1.aaaaaaaaoiusqhftxmiyjlulnxx5mdnqfv6pjx4hdcoks3exn7gsrcwpkpdq"
}

variable "instance_user_data" {
  default = ""
}

variable "subnet_ai_subnet_id" {
  default = ""
}

variable "security_list_id" {
  default = ""
}

variable "ssh_authorized_keys" {
  default = ""
}