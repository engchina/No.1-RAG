variable "availability_domain" {
  default = "bxtG:AP-TOKYO-1-AD-1"
}

variable "compartment_ocid" {
  default = ""
}

variable "adb_name" {
  default = "AIPOCADB"
}

variable "adb_password" {
  default = ""
}

variable "license_model" {
  default = ""
}

variable "adb_use_private_subnet" {
  description = "Whether to use a private subnet for Autonomous Database"
  type        = bool
  default     = false
}

variable "adb_subnet_id" {
  description = "Private subnet OCID for Autonomous Database (used when private subnet is enabled)"
  type        = string
  default     = ""
}

variable "instance_display_name" {
  default = "AIPOC_INSTANCE"
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

variable "subnet_ai_subnet_id" {
  default = ""
}

variable "ssh_authorized_keys" {
  default = ""
}
