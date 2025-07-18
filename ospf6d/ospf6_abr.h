// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (C) 2004 Yasuhiro Ohara
 */

#ifndef OSPF6_ABR_H
#define OSPF6_ABR_H

/* for struct ospf6_route */
#include "ospf6_route.h"
/* for struct ospf6_prefix */
#include "ospf6_proto.h"

#define OSPF6_ABR_TASK_DELAY 5

/* Debug option */
extern unsigned char conf_debug_ospf6_abr;
#define OSPF6_DEBUG_ABR_ON() (conf_debug_ospf6_abr = 1)
#define OSPF6_DEBUG_ABR_OFF() (conf_debug_ospf6_abr = 0)
#define IS_OSPF6_DEBUG_ABR (conf_debug_ospf6_abr)

#define OSPF6_ABR_SUMMARY_METRIC(E)                                            \
	(ntohl((E)->metric & htonl(OSPF6_EXT_PATH_METRIC_MAX)))
#define OSPF6_ABR_SUMMARY_METRIC_SET(E, C)                                     \
	{                                                                      \
		(E)->metric &= htonl(0x00000000);                              \
		(E)->metric |= htonl(OSPF6_EXT_PATH_METRIC_MAX) & htonl(C);    \
	}

#define OSPF6_ABR_RANGE_CLEAR_COST(range) (range->path.cost = OSPF_AREA_RANGE_COST_UNSPEC)
#define IS_OSPF6_ABR(o) ((o)->flag & OSPF6_FLAG_ABR)

extern bool ospf6_check_and_set_router_abr(struct ospf6 *o);

extern void ospf6_abr_enable_area(struct ospf6_area *oa);
extern void ospf6_abr_disable_area(struct ospf6_area *oa);

extern int ospf6_abr_originate_summary_to_area(struct ospf6_route *route,
					       struct ospf6_area *area);
extern void ospf6_abr_originate_summary(struct ospf6_route *route,
					struct ospf6 *ospf6);
extern void ospf6_abr_examin_summary(struct ospf6_lsa *lsa,
				     struct ospf6_area *oa);
extern void ospf6_abr_defaults_to_stub(struct ospf6 *ospf6);
extern void ospf6_abr_examin_brouter(uint32_t router_id,
				     struct ospf6_route *route,
				     struct ospf6 *ospf6);
extern void ospf6_abr_range_reset_cost(struct ospf6 *ospf6);

extern void ospf6_abr_task(struct ospf6 *ospf6);
extern void ospf6_schedule_abr_task(struct ospf6 *ospf6);
extern void ospf6_execute_abr_task(struct ospf6 *ospf6);

extern int config_write_ospf6_debug_abr(struct vty *vty);
extern void install_element_ospf6_debug_abr(void);
extern int ospf6_abr_config_write(struct vty *vty);
extern void ospf6_abr_old_route_remove(struct ospf6_lsa *lsa,
				       struct ospf6_route *old,
				       struct ospf6_route_table *table);
extern void ospf6_abr_old_path_update(struct ospf6_route *old_route,
				      struct ospf6_route *route,
				      struct ospf6_route_table *table);
extern void ospf6_abr_init(void);
extern void ospf6_abr_range_update(struct ospf6_route *range,
				   struct ospf6 *ospf6);
extern int ospf6_ls_origin_same(struct ospf6_path *o_path,
				struct ospf6_path *r_path);

#endif /*OSPF6_ABR_H*/
