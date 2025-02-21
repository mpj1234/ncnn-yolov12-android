pluginManagement {
	repositories {
		maven {
			this.url = uri("https://maven.aliyun.com/repository/public/")
		}
		maven {
			this.url = uri("https://maven.aliyun.com/repository/google/")
		}
		maven {
			this.url = uri("https://maven.aliyun.com/repository/jcenter/")
		}
		google {
			content {
				includeGroupByRegex("com\\.android.*")
				includeGroupByRegex("com\\.google.*")
				includeGroupByRegex("androidx.*")
			}
		}
		mavenCentral()
		gradlePluginPortal()
	}
}
dependencyResolutionManagement {
	repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
	repositories {
		google()
		mavenCentral()
	}
}

rootProject.name = "yolov12"
include(":app")
 