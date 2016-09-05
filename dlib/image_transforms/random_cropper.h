// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_RaNDOM_CROPPER_H_
#define DLIB_RaNDOM_CROPPER_H_

#include "random_cropper_abstract.h"
#include "../threads.h"
#include <mutex>
#include <vector>
#include "interpolation.h"
#include "../image_processing/full_object_detection.h"
#include "../rand.h"

namespace dlib
{
    class random_cropper
    {
        chip_dims dims = chip_dims(300,300);
        bool randomly_flip = true;
        double max_rotation_degrees = 30;
        double min_object_height = 0.25; // cropped object will be at least this fraction of the height of the image.
        double max_object_height = 0.7; // cropped object will be at most this fraction of the height of the image.
        double background_crops_fraction = 0.1;
    public:

        double get_background_crops_fraction (
        ) const { return background_crops_fraction; }

        void set_background_crops_fraction (
            double value
        )
        {
            DLIB_CASSERT(0 <= value && value < 1);
            background_crops_fraction = value;
        }

        const chip_dims& get_chip_dims(
        ) const { return dims; }

        void set_chip_dims (
            const chip_dims& dims_
        ) { dims = dims_; }

        void set_chip_dims (
            unsigned long rows,
            unsigned long cols
        ) { set_chip_dims(chip_dims(rows,cols)); }

        bool get_randomly_flip (
        ) const { return randomly_flip; }

        void set_randomly_flip (
            bool value
        ) { randomly_flip = value; }

        double get_max_rotation_degrees (
        ) const { return max_rotation_degrees; }
        void set_max_rotation_degrees (
            double value
        ) { max_rotation_degrees = std::abs(value); }

        double get_min_object_height (
        ) const { return min_object_height; }
        void set_min_object_height (
            double value
        ) 
        { 
            DLIB_CASSERT(0 < value && value < 1);
            min_object_height = value; 
        }

        double get_max_object_height (
        ) const { return max_object_height; }
        void set_max_object_height (
            double value
        ) 
        { 
            DLIB_CASSERT(0 < value && value < 1);
            max_object_height = value; 
        }

        template <
            typename array_type
            >
        void operator() (
            size_t num_crops,
            const array_type& images,
            const std::vector<std::vector<mmod_rect>>& rects,
            array_type& crops,
            std::vector<std::vector<mmod_rect>>& crop_rects
        )
        {
            DLIB_CASSERT(images.size() == rects.size());
            crops.resize(num_crops);
            crop_rects.resize(num_crops);
            parallel_for(0, num_crops, [&](long i) {
                (*this)(images, rects, crops[i], crop_rects[i]);
            });
        }

        template <
            typename array_type,
            typename image_type
            >
        void operator() (
            const array_type& images,
            const std::vector<std::vector<mmod_rect>>& rects,
            image_type& crop,
            std::vector<mmod_rect>& crop_rects
        )
        {
            DLIB_CASSERT(images.size() == rects.size());
            size_t idx;
            { std::lock_guard<std::mutex> lock(rnd_mutex);
                idx = rnd.get_random_64bit_number()%images.size();
            }
            (*this)(images[idx], rects[idx], crop, crop_rects);
        }

        template <
            typename image_type1,
            typename image_type2
            >
        void operator() (
            const image_type1& img,
            const std::vector<mmod_rect>& rects,
            image_type2& crop,
            std::vector<mmod_rect>& crop_rects
        )
        {
            chip_details crop_plan;
            bool should_flip_crop;
            make_crop_plan(img, rects, crop_plan, should_flip_crop);

            extract_image_chip(img, crop_plan, crop);
            const rectangle_transform tform = get_mapping_to_chip(crop_plan);

            const unsigned long min_object_height_absolute = std::round(min_object_height*crop_plan.rows);

            // copy rects into crop_rects and set ones that are outside the crop to ignore or
            // drop entirely as appropriate.
            crop_rects.clear();
            for (auto rect : rects)
            {
                // map to crop
                rect.rect = tform(rect.rect);

                // if the rect is at least partly in the crop
                if (get_rect(crop).intersect(rect.rect).area() != 0)
                {
                    // set to ignore if not totally in the crop or if too small.
                    if (!get_rect(crop).contains(rect.rect) || rect.rect.height() < min_object_height_absolute)
                        rect.ignore = true;

                    crop_rects.push_back(rect);
                }
            }

            // Also randomly flip the image
            if (should_flip_crop)
            {
                image_type2 temp;
                flip_image_left_right(crop, temp); 
                swap(crop,temp);
                for (auto&& rect : crop_rects)
                    rect.rect = impl::flip_rect_left_right(rect.rect, get_rect(crop));
            }
        }

    private:

        template <typename image_type1>
        void make_crop_plan (
            const image_type1& img,
            const std::vector<mmod_rect>& rects,
            chip_details& crop_plan,
            bool& should_flip_crop
        )
        {
            std::lock_guard<std::mutex> lock(rnd_mutex);
            rectangle crop_rect;
            if (has_non_ignored_box(rects) && rnd.get_random_double() >= background_crops_fraction)
            {
                auto rect = rects[randomly_pick_rect(rects)].rect;
                // perturb the location of the crop by a small fraction of the object's size.
                const point rand_translate = dpoint(rnd.get_double_in_range(-0.1,0.1)*rect.width(), 
                    rnd.get_double_in_range(-0.1,0.1)*rect.height());

                // perturb the scale of the crop by a fraction of the object's size
                const double rand_scale_perturb = rnd.get_double_in_range(min_object_height, max_object_height); 

                const long box_size = rect.height()/rand_scale_perturb;
                crop_rect = centered_rect(center(rect)+rand_translate, box_size, box_size);
            }
            else
            {
                crop_rect = make_random_cropping_rect_resnet(img);
            }
            should_flip_crop = randomly_flip && rnd.get_random_double() > 0.5;
            const double angle = rnd.get_double_in_range(-max_rotation_degrees, max_rotation_degrees)*pi/180;
            crop_plan = chip_details(crop_rect, dims, angle);
        }

        bool has_non_ignored_box (
            const std::vector<mmod_rect>& rects
        ) const
        {
            for (auto&& b : rects)
            {
                if (!b.ignore)
                    return true;
            }
            return false;
        }

        size_t randomly_pick_rect (
            const std::vector<mmod_rect>& rects
        ) 
        {
            DLIB_CASSERT(has_non_ignored_box(rects));
            size_t idx = rnd.get_random_64bit_number()%rects.size();
            while(rects[idx].ignore)
                idx = rnd.get_random_64bit_number()%rects.size();
            return idx;
        }

        template <typename image_type>
        rectangle make_random_cropping_rect_resnet(
            const image_type& img
        )
        {
            // figure out what rectangle we want to crop from the image
            double mins = 0.466666666, maxs = 0.875;
            auto scale = mins + rnd.get_random_double()*(maxs-mins);
            auto size = scale*std::min(img.nr(), img.nc());
            rectangle rect(size, size);
            // randomly shift the box around
            point offset(rnd.get_random_32bit_number()%(img.nc()-rect.width()),
                rnd.get_random_32bit_number()%(img.nr()-rect.height()));
            return move_rect(rect, offset);
        }


        std::mutex rnd_mutex;
        dlib::rand rnd;

    };

}

#endif // DLIB_RaNDOM_CROPPER_H_

